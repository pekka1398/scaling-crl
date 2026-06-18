"""CRL algorithm: training step, critic loss, actor loss.

Pure JAX functions. No globals, no defaults, no circular imports.
env is captured via closure, not passed as JIT argument.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from crl.networks import Actor, SA_encoder, G_encoder
from crl.types import Transition
from crl.buffer import TrajectoryUniformSamplingQueue
from evaluator import CrlEvaluator


def make_actor_step(actor, env, mode="stochastic", **kwargs):
    """Create an actor step function. env is captured via closure (not JIT arg).

    mode: "deterministic", "stochastic", or "multi_sample"
    For multi_sample, pass obs_dim, goal_start_idx, goal_end_idx, sa_encoder, g_encoder, K in kwargs.
    """
    if mode == "deterministic":
        def actor_step(training_state, env_state, extra_fields):
            means, _ = actor.apply(training_state.actor_state.params, env_state.obs)
            actions = nn.tanh(means)
            nstate = env.step(env_state, actions)
            return nstate, Transition(
                observation=env_state.obs, action=actions, reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": {x: nstate.info[x] for x in extra_fields}},
            )
        return actor_step

    elif mode == "stochastic":
        def actor_step(training_state, env_state, key, extra_fields):
            means, log_stds = actor.apply(training_state.actor_state.params, env_state.obs)
            stds = jnp.exp(log_stds)
            actions = nn.tanh(means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype))
            nstate = env.step(env_state, actions)
            return nstate, Transition(
                observation=env_state.obs, action=actions, reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": {x: nstate.info[x] for x in extra_fields}},
            )
        return actor_step

    elif mode == "multi_sample":
        return make_multi_sample_actor_step(
            actor, kwargs["sa_encoder"], kwargs["g_encoder"], env,
            kwargs["obs_dim"], kwargs["goal_start_idx"], kwargs["goal_end_idx"], kwargs["K"])

    else:
        raise ValueError(f"Unknown actor mode: {mode}")


def make_multi_sample_actor_step(actor, sa_encoder, g_encoder, env, obs_dim, goal_start_idx, goal_end_idx, K):
    """Create a multi-sample actor step function.

    Samples K actions, picks best via Q-value (contrastive distance).
    env, obs_dim, goal indices captured via closure.
    """

    def multi_sample_actor_step(training_state, env_state, key, extra_fields):
        keys = jax.random.split(key, K)
        means, log_stds = actor.apply(training_state.actor_state.params, env_state.obs)
        stds = jnp.exp(log_stds)
        actions = jnp.stack([
            nn.tanh(means + stds * jax.random.normal(k, shape=means.shape, dtype=means.dtype))
            for k in keys
        ])

        state = env_state.obs[:, :obs_dim]
        goal = env_state.obs[:, obs_dim:]

        sa_reprs = jax.vmap(
            lambda a: sa_encoder.apply(training_state.critic_state.params["sa_encoder"], state, a)
        )(actions)
        g_repr = g_encoder.apply(training_state.critic_state.params["g_encoder"], goal)

        q_values = -jnp.sqrt(jnp.sum((sa_reprs - g_repr) ** 2, axis=-1))
        best_action_idx = jnp.argmax(q_values, axis=0)
        best_actions = jnp.take_along_axis(actions, best_action_idx[None, :, None], axis=0)[0]

        nstate = env.step(env_state, best_actions)
        return nstate, Transition(
            observation=env_state.obs, action=best_actions, reward=nstate.reward,
            discount=1 - nstate.done,
            extras={"state_extras": {x: nstate.info[x] for x in extra_fields}},
        )

    return multi_sample_actor_step


def make_get_experience(actor_step_fn, replay_buffer, unroll_length):
    """Create get_experience. unroll_length captured via closure."""

    @jax.jit
    def get_experience(training_state, env_state, buffer_state, key):
        def f(carry, unused_t):
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            env_state, transition = actor_step_fn(training_state, env_state, current_key,
                                                  extra_fields=("truncation", "seed"))
            return (env_state, next_key), transition

        (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=unroll_length)
        buffer_state = replay_buffer.insert(buffer_state, data)
        return env_state, buffer_state

    return get_experience


def make_prefill(get_experience, env_steps_per_actor_step):
    """Create prefill_replay_buffer using jax.lax.scan (not Python loop)."""

    def prefill_replay_buffer(training_state, env_state, buffer_state, key, num_prefill_actor_steps):
        def f(carry, unused):
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            env_state, buffer_state = get_experience(training_state, env_state, buffer_state, key)
            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step)
            return (training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(f, (training_state, env_state, buffer_state, key),
                            (), length=num_prefill_actor_steps)[0]

    return prefill_replay_buffer


def make_update_actor(actor, sa_encoder, g_encoder, target_entropy, args):
    """Create actor + alpha update. target_entropy and args captured via closure."""

    @jax.jit
    def update_actor_and_alpha(transitions, training_state, key):
        actor_batch_size = args.batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[:actor_batch_size], transitions)

        def actor_loss(actor_params, critic_params, log_alpha, transitions, key):
            obs = transitions.observation
            state = obs[:, :args.obs_dim]
            future_state = transitions.extras["future_state"]
            goal = future_state[:, args.goal_start_idx:args.goal_end_idx]
            observation = jnp.concatenate([state, goal], axis=1)

            means, log_stds = actor.apply(actor_params, observation)
            stds = jnp.exp(log_stds)
            x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
            action = nn.tanh(x_ts)
            log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
            log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
            log_prob = log_prob.sum(-1)

            sa_repr = sa_encoder.apply(critic_params["sa_encoder"], state, action)
            g_repr = g_encoder.apply(critic_params["g_encoder"], goal)
            qf_pi = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))

            if args.disable_entropy:
                loss = -jnp.mean(qf_pi)
            else:
                loss = jnp.mean(jnp.exp(log_alpha) * log_prob - qf_pi)
            return loss, log_prob

        def alpha_loss(alpha_params, log_prob):
            alpha = jnp.exp(alpha_params["log_alpha"])
            return jnp.mean(alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - target_entropy)))

        (actorloss, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
            training_state.actor_state.params, training_state.critic_state.params,
            training_state.alpha_state.params['log_alpha'], transitions, key)
        new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

        alphaloss, alpha_grad = jax.value_and_grad(alpha_loss)(
            training_state.alpha_state.params, log_prob)
        new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

        training_state = training_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)
        metrics = {
            "sample_entropy": -log_prob, "actor_loss": actorloss, "alpha_loss": alphaloss,
            "log_alpha": training_state.alpha_state.params["log_alpha"],
        }
        return training_state, metrics

    return update_actor_and_alpha


def make_update_critic(sa_encoder, g_encoder, args):
    """Create critic update. args captured via closure."""

    @jax.jit
    def update_critic(transitions, training_state, key):
        critic_batch_size = args.batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[:critic_batch_size], transitions)

        def critic_loss(critic_params, transitions, key):
            sa_encoder_params = critic_params["sa_encoder"]
            g_encoder_params = critic_params["g_encoder"]
            obs = transitions.observation[:, :args.obs_dim]
            action = transitions.action
            sa_repr = sa_encoder.apply(sa_encoder_params, obs, action)
            g_repr = g_encoder.apply(g_encoder_params, transitions.observation[:, args.obs_dim:])

            logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
            loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
            logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
            loss += args.logsumexp_penalty_coeff * jnp.mean(logsumexp ** 2)

            logits_pos = jnp.diag(logits)
            mask = 1.0 - jnp.eye(logits.shape[0])
            logits_neg_mean = (logits * mask).sum() / mask.sum()
            correct = jnp.argmax(logits, axis=1) == jnp.arange(logits.shape[0])
            return loss, (logsumexp, correct, logits_pos, logits_neg_mean)

        (loss, (logsumexp, correct, logits_pos, logits_neg_mean)), grad = jax.value_and_grad(critic_loss, has_aux=True)(
            training_state.critic_state.params, transitions, key)
        new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
        training_state = training_state.replace(critic_state=new_critic_state)
        metrics = {
            "categorical_accuracy": jnp.mean(correct.astype(jnp.float32)),
            "logits_pos": logits_pos.mean(), "logits_neg": logits_neg_mean,
            "logsumexp": logsumexp.mean(), "critic_loss": loss,
        }
        return training_state, metrics

    return update_critic


def make_sgd_step(update_actor_and_alpha, update_critic):
    @jax.jit
    def sgd_step(carry, transitions):
        training_state, key = carry
        key, critic_key, actor_key = jax.random.split(key, 3)
        training_state, actor_metrics = update_actor_and_alpha(transitions, training_state, actor_key)
        training_state, critic_metrics = update_critic(transitions, training_state, critic_key)
        training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)
        return (training_state, key), {**actor_metrics, **critic_metrics}
    return sgd_step


def make_training_step(sgd_step, get_experience, replay_buffer, args):
    """args captured via closure."""

    @jax.jit
    def training_step(training_state, env_state, buffer_state, key, t=None):
        experience_key1, experience_key2, sampling_key, training_key, sgd_batches_key = jax.random.split(key, 5)

        env_state, buffer_state = get_experience(training_state, env_state, buffer_state, experience_key1)
        training_state = training_state.replace(
            env_steps=training_state.env_steps + args.env_steps_per_actor_step)

        transitions_list = []
        for _ in range(args.num_episodes_per_env):
            buffer_state, new_transitions = replay_buffer.sample(buffer_state)
            transitions_list.append(new_transitions)

        transitions = jax.tree_util.tree_map(
            lambda *arrays: jnp.concatenate(arrays, axis=0), *transitions_list)

        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0))(
            (args.gamma, args.obs_dim, args.goal_start_idx, args.goal_end_idx), transitions, batch_keys)

        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions)
        permutation = jax.random.permutation(experience_key2, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)

        num_full_batches = len(transitions.observation) // args.batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[:num_full_batches * args.batch_size], transitions)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, args.batch_size) + x.shape[1:]), transitions)

        if args.use_all_batches == 0:
            num_total_batches = transitions.observation.shape[0]
            selected_indices = jax.random.permutation(sgd_batches_key, num_total_batches)[:args.num_sgd_batches_per_training_step]
            transitions = jax.tree_util.tree_map(lambda x: x[selected_indices], transitions)

        (training_state, _), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)
        return (training_state, env_state, buffer_state), metrics

    return training_step


def make_training_epoch(training_step, replay_buffer, args):
    """args captured via closure."""

    @jax.jit
    def training_epoch(training_state, env_state, buffer_state, key):
        def f(carry, t):
            ts, es, bs, k = carry
            k, train_key = jax.random.split(k, 2)
            (ts, es, bs), metrics = training_step(ts, es, bs, train_key, t)
            return (ts, es, bs, k), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f, (training_state, env_state, buffer_state, key),
            jnp.arange(args.num_training_steps_per_epoch * args.training_steps_multiplier))

        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics

    return training_epoch


def setup_evaluator(actor, sa_encoder, g_encoder, eval_env, obs_dim, eval_actor_key, args, eval_env_key):
    """Setup evaluator. eval_env and eval_actor_key captured via closure in actor steps."""

    if args.eval_actor == 0:
        step_fn = make_actor_step(actor, eval_env, mode="deterministic")
        return CrlEvaluator(
            lambda ts, env, es, extra_fields: step_fn(ts, es, extra_fields),
            eval_env, num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length, key=eval_env_key)

    elif args.eval_actor == 1:
        # Use eval_actor_key (derived from main key) for reproducibility
        # fold_in with env_steps so different epochs get different keys
        def eval_step(ts, env, es, extra_fields):
            k = jax.random.fold_in(eval_actor_key, ts.env_steps)
            means, log_stds = actor.apply(ts.actor_state.params, es.obs)
            stds = jnp.exp(log_stds)
            actions = nn.tanh(means + stds * jax.random.normal(k, shape=means.shape, dtype=means.dtype))
            nstate = env.step(es, actions)
            return nstate, Transition(
                observation=es.obs, action=actions, reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": {x: nstate.info[x] for x in extra_fields}},
            )
        return CrlEvaluator(eval_step, eval_env, num_eval_envs=args.num_eval_envs,
                            episode_length=args.episode_length, key=eval_env_key)

    elif args.eval_actor > 1:
        K = args.eval_actor
        multi_step = make_multi_sample_actor_step(
            actor, sa_encoder, g_encoder, eval_env,
            obs_dim, args.goal_start_idx, args.goal_end_idx, K)
        _eval_actor_key = eval_actor_key
        def eval_step(ts, env, es, extra_fields):
            key = jax.random.fold_in(_eval_actor_key, ts.env_steps)
            nstate, transition = multi_step(ts, es, key, extra_fields)
            return nstate, transition
        return CrlEvaluator(eval_step, eval_env, num_eval_envs=args.num_eval_envs,
                            episode_length=args.episode_length, key=eval_env_key)

    else:
        raise ValueError(f"eval_actor={args.eval_actor} not supported")
