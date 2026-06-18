 ### 1. PureJaxRL — 最簡，單檔實現                                                                                       
                                                                                                                         
 - 整個 PPO 就一個檔案（ppo_continuous_action.py ~300 行）                                                               
 - Config 就是 Python dict，沒有任何 framework                                                                           
 - 沒有 checkpoint、沒有 logger、沒有 SLURM                                                                              
 - 整個 training loop 是一個 jax.jit 的 function                                                                         
 - 適合：學習 JAX RL 怎麼寫，不適合當 infra                                                                              
                                                                                                                         
 ### 2. JaxRL (ikostrikov) — 乾淨的 agent/learner 分離                                                                   
                                                                                                                         
 - 核心設計：Model class 把 params + apply_fn + optimizer 包在一起                                                       
 - 每個算法一個目錄：agents/sac/, agents/ddpg/, etc.                                                                     
 - SAC learner 只有 ~120 行，非常乾淨                                                                                    
 - Config 用 Python dataclass（sac_default.py）                                                                          
 - 沒有 checkpoint、沒有 SLURM                                                                                           
 - 適合：參考它的 Model 抽象和 agent 目錄結構                                                                            
                                                                                                                         
 ### 3. Stoix (InstaDeep) — 最完整的 JAX RL infra                                                                        
                                                                                                                         
 - 用 Orbax checkpointing（跟我們一樣）                                                                                  
 - 用 Hydra config（層級式 YAML）                                                                                        
 - 用 flashbax 做 replay buffer                                                                                          
 - 用 stoa 做 env wrapper                                                                                                
 - jax.pmap 多 GPU                                                                                                       
 - SLURM launcher 內建                                                                                                   
 - 適合：參考它的 checkpoint 和 logger 設計                                                                              
                                                                                                                         
 ### 4. Mava (InstaDeep) — Stoix 的 multi-agent 版                                                                       
                                                                                                                         
 - checkpointing code 跟 Stoix 幾乎一樣（同一個團隊）                                                                    
 - 多了 GNN、Attention 網路                                                                                              
 - 不適合你：MARL 用的                                                                                                   
                                                                                                                         
 ### 5. RLax (DeepMind) — 純建構塊                                                                                       
                                                                                                                         
 - 只有 RL 數學運算：Q-learning ops、policy gradient ops、distributions                                                  
 - 不是框架，是 library                                                                                                  
 - 適合：如果你需要自定義 loss function，可以從這裡拿                                                                    
                                                                                                                         
 ### 6. Acme (DeepMind) — 太重                                                                                           
                                                                                                                         
 - 複雜的架構：adder、dataset、environment loop、inference server                                                        
 - 已經不太維護                                                                                                          
 - 不推薦                                                                                                                
                                                                                                                         
 ### 7. Dopamine (Google) — Discrete RL                                                                                  
                                                                                                                         
 - 有自己的 checkpointing（Orbax + msgpack）                                                                             
 - 主要是 DQN/C51/Rainbow                                                                                                
 - 不適合你：continuous control 用的                                                                                     
                    

整理一下目前所有決策：                                                                                              
                                                                                                                         
 要做的事                                                                                                                
                                                                                                                         
 ### A. 重構 train.py                                                                                                    
                                                                                                                         
 - 拆成獨立模組：crl/networks.py、crl/algorithm.py、utils/checkpoint.py、utils/env_factory.py、utils/logging.py          
 - Training step 從 train.py 拆出來，做成純 function                                                                     
 - jax.lax.scan 在 training step 層級，epoch loop 保留 Python 控制流                                                     
 - 消除所有硬編碼和 fallback — 沒有 default values，YAML 沒有就報錯                                                      
                                                                                                                         
 ### B. 每個算法獨立目錄（crl/）                                                                                         
                                                                                                                         
 - CRL 的 networks、algorithm、buffer 都在 crl/ 裡                                                                       
 - 以後加新算法就加新目錄                                                                                                
                                                                                                                         
 ### C. Checkpointer 參數從 YAML 讀                                                                                      
                                                                                                                         
 - save_interval_epochs、max_to_keep、keep_period 全部從 config 來                                                       
 - 不設 default values — YAML 裡沒有就直接報錯，不要 fallback                                                            
                                                                                                                         
 ### D. Checkpointable protocol                                                                                          
                                                                                                                         
 - 定義 to_state_dict / from_state_dict                                                                                  
 - 支援 optax optimizer state（Orbax 的 PyTreeCheckpointer 可以存）                                                      
                                                                                                                         
 ### E. flashbax replay buffer                                                                                           
                                                                                                                         
 - 換掉自寫的 buffer                                                                                                     
                                                                                                                         
 ### F. jax.pmap 多 GPU                                                                                                  
                                                                                                                         
 - 從 Stoix 學                                                                                                           
                                                                                                                         
 不做的事                                                                                                                
                                                                                                                         
 ┌─────────────────────────────────────────────────────────────┬───────────────────────────┐                             
 │ 項目                                                        │ 原因                      │                             
 ├─────────────────────────────────────────────────────────────┼───────────────────────────┤                             
 │ Model 抽象（JaxRL 的 params + apply_fn + optimizer 包一起） │ 不需要                    │                             
 ├─────────────────────────────────────────────────────────────┼───────────────────────────┤                             
 │ best model tracking（Orbax 的 best_fn）                     │ 不需要                    │                             
 ├─────────────────────────────────────────────────────────────┼───────────────────────────┤                             
 │ Hydra config                                                │ 保持 YAML + dataclass     │                             
 ├─────────────────────────────────────────────────────────────┼───────────────────────────┤                             
 │ TensorBoard                                                 │ 只用 Wandb                │                             
 ├─────────────────────────────────────────────────────────────┼───────────────────────────┤                             
 │ 全 JIT training loop（PureJaxRL 模式）                      │ side effects 太多，不適合 │                             
 ├─────────────────────────────────────────────────────────────┼───────────────────────────┤                             
 │ Acme、Dopamine、RLax、Mava                                  │ 不適合或太重              │                             
 └─────────────────────────────────────────────────────────────┴───────────────────────────┘                             
                                                                                                                         
 優先順序                                                                                                                
                                                                                                                         
 1. A + B — 重構 train.py + 拆出 crl/ 目錄（最大價值）                                                                   
 2. C + D — Checkpointer config + Checkpointable protocol（小改動）                                                      
 3. E — flashbax（中等改動）                                                                                             
 4. F — pmap 多 GPU（最後做，目前單 GPU 夠用） 