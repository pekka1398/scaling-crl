Go to reinforcementlearning
r/reinforcementlearning
•
6mo ago
thecity2

1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities
arxiv.org

Open
This was an award winning paper at NeurIPS this year.

Scaling up self-supervised learning has driven breakthroughs in language and vision, yet comparable progress has remained elusive in reinforcement learning (RL). In this paper, we study building blocks for self-supervised RL that unlock substantial improvements in scalability, with network depth serving as a critical factor. Whereas most RL papers in recent years have relied on shallow architectures (around 2 - 5 layers), we demonstrate that increasing the depth up to 1024 layers can significantly boost performance. Our experiments are conducted in an unsupervised goal-conditioned setting, where no demonstrations or rewards are provided, so an agent must explore (from scratch) and learn how to maximize the likelihood of reaching commanded goals. Evaluated on simulated locomotion and manipulation tasks, our approach increases performance on the self-supervised contrastive RL algorithm by 2× - 50×, outperforming other goal-conditioned baselines. Increasing the model depth not only increases success rates but also qualitatively changes the behaviors learned.


Upvote
30

Downvote

28
Go to comments


Share
u/OutlierAI_Official avatar
OutlierAI_Official
•
Promoted

Earn up to $50 USD per hour. Fully remote. Free model usage.
Apply Now
outlier.ai
Thumbnail image: Earn up to $50 USD per hour. Fully remote. Free model usage.
Join the conversation
Sort by:

Best

Search Comments
Expand comment search
Comments Section
u/CaseFlatline avatar
CaseFlatline
•
6mo ago
•
Edited 6mo ago
One of the top 3 papers. The others are listed here along with runners up: https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/

and comments for the RL paper: https://openreview.net/forum?id=s0JVsx3bx1


Upvote
4

Downvote

Reply

Award

Share

u/blimpyway avatar
blimpyway
•
6mo ago
100000 layers is way bigger.



Upvote
4

Downvote

Reply

Award

Share

thecity2
OP
•
6mo ago
100000 lawyers is way bigger


Upvote
4

Downvote

Reply

Award

Share

u/b_eysenbach avatar
b_eysenbach
•
6mo ago
Author of the paper here. Happy to answer any questions about the paper!

Responding to a few questions raised so far in the discussion:

> more layers
One of the misconceptions about the paper is that throwing more layers at any RL algorithms should boost performance. That's not the case. Rather, one of the key findings was that scaling depth required using a particular learning rule, one more akin to self-supervised learning than reinforcement learning.

> how much the result depends more on layers for computational steps or for parameters
@radarsat1 I think that's spot on! The observations here aren't that high-dimensional. So it really does seem like the additional capacity is being used for a sort of "reasoning" rather than just compressing high-dimensional observation. We spent some time experimenting with weight tying / recurrent versions and couldn't get it to work, but I think that it should be possible to significantly decrease the parameter count while still making use of a large amount of computation.



Upvote
4

Downvote

Reply

Award

Share

thecity2
OP
•
6mo ago
Hey thanks for posting here. I literally tried to "throw more layers" at a model I'm working on after I read the paper...alas I can report it did not get better haha. Worth a shot though.



Upvote
1

Downvote

Reply

Award

Share

u/b_eysenbach avatar
b_eysenbach
•
6mo ago
Depending on the application, you should try changing the objective! It's arguably simpler than the PPO/SAC/TD3/etc objective you're likely currently using.



Upvote
1

Downvote

Reply

Award

Share

thecity2
OP
•
6mo ago
Could CRL work for a zero-sum game like basketball? I'm building a 2D "hex world" version of basketball called Basket World. I'm using PPO (SB3) currently. It's definitely learning something, but very sample inefficient. If you have time or interest take a look (there are some gifs that show "game play"). https://github.com/EvanZ/basketworld



Upvote
1

Downvote

Reply

Award

Share

u/b_eysenbach avatar
b_eysenbach
•
6mo ago
You could give it a shot!
We've recently found that these methods work fairly well at getting teams of agents to coordinate (e.g., in starcraft like tasks): https://chirayu-n.github.io/gcmarl
The problems we've looked at, though, have been cooperative (not two-player zero-sum).



Upvote
3

Downvote

Reply

Award

Share

thecity2
OP
•
6mo ago
>We reframe this problem instead as a goal-reaching problem: we give the agents a shared goal and let them figure out how to cooperate and reach that goal without any additional guidance. The agents do this by learning how to maximize the likelihood of visiting this shared goal.

Interesting, thanks. Indeed this is exactly what I try to do in my model. The reward on offense is simply the expected shot value, which encourages better shots. And the defense has the inverse goal, to stop the offense from getting good shots. The way you framed the problem seems exactly suited to my case.


Upvote
1

Downvote

Reply

Award

Share

u/Meshyai avatar
u/Meshyai
•
Promoted

Spent 4 hours sculpting a mini in ZBrush last week. Then tried the same thing in Meshy. 90 seconds. The quality gap is closing fast.
View More
meshy.ai
Thumbnail image: Spent 4 hours sculpting a mini in ZBrush last week. Then tried the same thing in Meshy. 90 seconds. The quality gap is closing fast.
u/gerryflap avatar
gerryflap
•
6mo ago
MORE LAYERS!!!!1!

I really like this paper though. I haven't been following RL that much for a few years but the explanations and math were easy enough to follow to get the gist of it. If I find the time and energy (tm) I might try to implement this and throw it onto some environments.



Upvote
6

Downvote

Reply

Award

Share

dekiwho
•
6mo ago
only works on 2 algos, and only very good on 1 algo.... there are some flaws highlighted in the open review...



Upvote
-1

Downvote

Reply

Award

Share

hunted7fold
•
6mo ago
I think you’re missing the point. It’s not that the scaling formula only works on 1 algo. It’s that the one algo scales. The goal is to find a scalable RL method, and this paper is showing that it’s CRL. It’s not to show a new architecture, it’s to show CRL is scalable



Upvote
4

Downvote

Reply

Award

Share

u/Witty-Elk2052 avatar
Witty-Elk2052
•
6mo ago
think this paper exposes just how deficient in representation learning the other RL algorithms are, in particular SAC


Upvote
2

Downvote

Reply

Award

Share

dekiwho
•
6mo ago
I am not missing any point.

You literally saying what I said with different words.

They dont fully compare rainbow, dqn, tdmpc, dreamerv3, r2d2,r2d4,Simba, SimbaV2 etcc..... this paper is not robust. There are 100s if not thousands of RL algo variant.

LIke why didnt they compare c51? A much more common algo and familiar to people? It too uses cross entropy . Did we really need to pull CRL out of the dead for this ?

Algos been scalable for a decade now... lol people living under a rock ?

Scaling RL nets is nothing new, it would be new if they could achieve the same performance of 1000 layers with 10 layers, that any person can run on consumer grade hardware



Upvote
-2

Downvote

Reply

Award

Share

thecity2
OP
•
6mo ago
“All truth passes through three stages: First, it is ridiculed; Second, it is violently opposed; Third, it is accepted as being self-evident.”

Congrats on progressing so quickly to stage 3.



Upvote
1

Downvote

Reply

Award

Share

dekiwho
•
6mo ago
Read my last paragraph again



Upvote
1

Downvote

Reply

Award

Share

thecity2
OP
•
6mo ago
Can you cite the RL papers for a decade that have used 1000 layers like this? I’m sure interested to read about it.


Upvote
2

Downvote

Reply

Award

Share

More replies
u/TemporaryTight1658 avatar
TemporaryTight1658
•
6mo ago
It probably remembers better all states.

Therefore have better benchmarks ?


Upvote
2

Downvote

Reply

Award

Share

timelyparadox
•
6mo ago
Mathematically i do not see how these layers are actually encoding any additional information



Upvote
-1

Downvote

Reply

Award

Share

u/radarsat1 avatar
radarsat1
•
6mo ago
I definitely found myself wondering as I read it how much the result depends more  on layers for computational steps or for parameters. In other words I'd love to see this compared with a recursive approach where the same layers are executed many times.


Upvote
2

Downvote

Reply

Award

Share

u/Vegetable-Result-577 avatar
Vegetable-Result-577
•
6mo ago
Well, they do. More layers means more activations, more activations - more correlation explained. It's still throwing more gpus to solve 2*2 instead of a paradigm shift, but there's still some margin left in this mechanics, and nvidia wont ath without such papers



Upvote
1

Downvote

Reply

Award

Share

timelyparadox
•
6mo ago
Thats not entirely true, mathematically there is diminishing returns



Upvote
1

Downvote

Reply

Award

Share

u/Vegetable-Result-577 avatar
Vegetable-Result-577
•
6mo ago
That's not exactly true, mathematically deep layer nesting leads to better data representation, with the point of diminishing returns being a function of data entropy.

Upd: how can you not get it, broo, just add more layers and vibe code, duh!


Upvote
1

Downvote

Reply

Award

Share

dekiwho
•
6mo ago
likewise, and only works nicely on 1 algo and limited on another. so its meh .

Clickbait title


NeurIPS 2025 Best Paper | 1000 Layer Networks for Self-Supervised RL
Uno Whoiam
Uno Whoiam
疑问请付费咨询~ AI&CV/科技人文求知者/好读书欲求甚解
​关注他
收录于 · 人工智能与计算机视觉笔记
内容疑似 AI 生成
4 人赞同了该文章
​
目录
收起
1. 核心思想一句话总结 (Elevator Pitch)
2. 论文背景与动机 (Background & Motivation)
2.1 试图解决的问题
2.2 为什么重要？
2.3 之前的局限性
3. 核心方法/模型详解 (The Core Method/Model)
3.1 总体框架：CRL + ResNet
3.2 关键技术点剖析
A. 算法核心：对比强化学习 (Contrastive RL, CRL)
B. 架构核心：ResNets for RL
4. 实验与结果分析 (Experiments & Results)
4.1 核心结果：深度带来质变
4.2 涌现行为 (Emergent Behaviors)
4.3 为什么深度有效？ (Why Scaling Happens)
5. 论文的贡献与影响 (Contribution & Impact)
5.1 主要贡献
5.2 潜在影响
5.3 未来方向
6. 结论 (Conclusion)
在深度学习的Scaling Law（缩放定律）统治下，NLP和CV早已迈入了大模型时代。然而，强化学习（RL）似乎被遗忘在了角落。

长期以来，RL社区面临一个令人困惑的现象：深度反而是累赘。主流的SOTA算法（如SAC, TD3）通常只使用2到4层的MLP（多层感知机）。一旦网络加深，训练就会变得极不稳定，性能不升反降。

NeurIPS 2025的这篇最佳论文《1000 Layer Networks for Self-Supervised RL》彻底打破了这一成见。普林斯顿大学的研究团队证明，通过正确的算法范式转换与架构设计，RL不仅可以扩展到1000层，还能涌现出浅层网络无法企及的拓扑理解能力。



论文标题：1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities

来源：NeurIPS 2025 (arXiv:2503.14858v3)

关键词：Self-Supervised RL, Scaling Laws, Deep ResNets, Goal-Conditioned

https://arxiv.org/pdf/2503.14858

1. 核心思想一句话总结 (Elevator Pitch)
这篇论文证明了在无监督目标导向（Goal-Conditioned） 的设置下，结合对比学习（Contrastive Learning） 与残差网络（ResNet） 架构，可以将强化学习网络的深度成功扩展至 1000层以上，从而实现性能的爆发式增长（2倍-50倍）并涌现出全新的复杂智能行为。


2. 论文背景与动机 (Background & Motivation)
2.1 试图解决的问题
在深度学习的“黄金时代”，CV和NLP都遵循着“Scaling Laws”（缩放定律）：模型越大、层数越深，性能越强，甚至会涌现出（Emergent）小模型不具备的能力。 然而，强化学习（RL）一直是这个定律的“弃儿”。长期以来，RL社区面临一个痛点：

浅层诅咒：主流RL模型通常只有2-4层MLP。
缩放失效：一旦简单增加深度，训练就会变得极不稳定，或者性能饱和。
2.2 为什么重要？
如果RL不能Scaling（扩增规模），它就永远无法处理像通用机器人控制、复杂策略博弈等需要高度抽象理解的现实世界问题。现有的RL更像是“死记硬背”的浅层反应，缺乏深层推理能力。

2.3 之前的局限性
信号稀疏：传统RL依赖标量奖励（Scalar Reward）。对于深层网络来说，仅仅靠一个稀疏的奖励信号来调整上亿个参数，梯度的“信噪比”太低。
架构陈旧：很多RL工作还在使用原始的MLP，缺乏像CV中BatchNorm、ResBlock这样成熟的深层训练稳定技术。
3. 核心方法/模型详解 (The Core Method/Model)
这篇论文的成功并非发明了全新的算法，而是发现了一个“黄金配方”，将现有的技术积木巧妙组合，打通了RL的任督二脉。

3.1 总体框架：CRL + ResNet
论文的核心策略可以概括为：自监督对比强化学习（CRL） + 现代深度架构。

3.2 关键技术点剖析
A. 算法核心：对比强化学习 (Contrastive RL, CRL)
这部分对CV博士来说非常亲切。

传统RL：通过预测一个具体的Q值（数值）来评估动作好坏。
CRL（本论文方法）：不直接回归Q值，而是像SimCLR或MoCo一样学习状态空间和目标空间的表征（Representation）。
比喻：想象你在玩迷宫。传统RL试图记住每一步离终点还有“几米”（回归数值）。CRL则是学习一张“地图”，它通过对比学习知道：状态A和目标G是属于同一条可行路径的（正样本），而状态A和随机目标G’是无关的（负样本）。


数学原理： Critic（评论家网络）的目标是最小化InfoNCE损失： 

其中 
。
：当前状态-动作的Embedding。
：目标的Embedding。
直觉：如果 
 能到达 
，就拉近它们的距离；否则推远。这种分类/度量学习的任务比回归Q值提供了更密集的梯度信号，适合训练深层网络。


B. 架构核心：ResNets for RL

为了让网络能堆到1000层，作者引入了CV中的标准操作，但在RL中经过了精细调优：

残差连接 (Residual Connections)：
。
作用：防止梯度消失，让深层网络至少不比浅层差（恒等映射）。
层归一化 (LayerNorm)：
作用：稳定每一层的激活分布，这对RL这种数据分布随策略不断变化的场景至关重要。
Swish 激活函数：
相比ReLU，Swish (
) 更平滑，有助于梯度的流动。
创新点总结：作者发现，单独使用CRL或者单独加深网络都不行。只有当“高密度的自监督信号（CRL）” 遇到 “能承载深度的架构（ResNet）” 时，RL的Scaling才真正发生。

4. 实验与结果分析 (Experiments & Results)
作者在JaxGCRL代码库（基于Brax/MJX物理引擎）上进行了大规模实验，涵盖了从简单的机械臂推物到高难度的Humanoid迷宫导航。

4.1 核心结果：深度带来质变
碾压基线：在10个环境中的8个，Scaling后的CRL击败了所有基线（包括SAC, TD3, GCBC等）。
临界点现象 (Phase Transitions)：
性能不是线性增长的，而是像台阶一样。
例如在Humanoid U-Maze任务中，深度从4层加到8层没反应，但到了64层突然爆发，到了256层又涌现出新能力。
数据支撑：在Humanoid Big Maze任务中，64层网络的性能是4层网络的 1051倍（见论文Table 1）。

4.2 涌现行为 (Emergent Behaviors)
这是论文最精彩的部分之一。

浅层网络 (Depth 4)：控制的人形机器人像个醉汉，只是简单地把自己“扔”向目标，然后摔倒。
中层网络 (Depth 16)：学会了直立行走。
深层网络 (Depth 64-256)：面对只有通过“钻过去”才能到达的目标，深层Agent学会了做体操——先下蹲、折叠身体，利用杠杆原理翻越障碍物（如图3所示）。
解读：这说明深度不仅仅提高了成功率，还改变了策略的定性（Qualitative） 本质。
4.3 为什么深度有效？ (Why Scaling Happens)
作者通过可视化Q值图（Heatmap）给出了极具说服力的解释（见图9）：

浅层网络：学到的是欧几里得距离。即使中间有墙，它也认为墙后的目标很“近”。这导致Agent一直撞墙。
深层网络：学到了测地线距离（Geodesic Distance）。Q值热力图完美地沿着迷宫的走廊弯曲。这意味着深层网络理解了环境的拓扑结构 (Topology)。

5. 论文的贡献与影响 (Contribution & Impact)
5.1 主要贡献
打破成见：证明了RL并非不能Scaling，而是以前的方法（算法+架构）没对齐。
具体路径：给出了一套可复现的配方（CRL + ResNet + LayerNorm + Swish），成功训练了1000层的RL智能体。
机理解析：揭示了深度网络在RL中的作用是更好地建模环境的拓扑结构和实现“经验缝合（Stitching）”。
5.2 潜在影响
RL的“GPT时刻”前夜：这篇论文预示着，如果我们能提供足够好的自监督信号和算力，RL模型也可以像LLM一样无限扩展。
从专用到通用：未来的机器人大脑可能不再是针对特定任务训练的小模型，而是一个预训练的、深层的、理解物理世界拓扑结构的通用大模型。
5.3 未来方向
离线微调：目前还是在线（Online）训练，成本高。未来如何将这些深层模型用于Offline RL？
Transformer化：本文用了MLP ResNet，下一步必然是探索Transformer架构在RL Scaling中的上限。
6. 结论 (Conclusion)
这篇论文可以理解为RL领域的“ResNet时刻”。

作者通过抛弃传统的标量奖励回归，转而拥抱对比学习（即CV中的度量学习），并结合深层残差网络，成功让强化学习模型突破了“浅层瓶颈”。这不仅带来了性能的数倍提升，更重要的是，它让智能体“涌现”出了理解空间拓扑和复杂运动规划的高级智慧。这篇工作强有力地提示我们：Self-Supervision 是通往大规模强化学习的必经之路。 