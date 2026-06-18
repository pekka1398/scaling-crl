1. config 超參數跟實驗的名字跟事項的資料結構 實驗的超參數跟數據來源統一 其他地方只能調用 不能每個地方(log名字 job裡實驗名字 等等都自己重新組裝一遍) 種子碼與確定性」管理
2. checkpoint跟resume
 " 使用異步保存機制（Background thread 寫入），或者使用類似 TensorBoard / WandB 只傳指標，模型權重限制保存數量（只留最新和最好 你的訓練腳本必須具備容錯能力（Fault Tolerance）。

做法： 引入 PyTorch Elastic (torchrun)。確保代碼在啟動時會自動檢查是否存在最新的 Checkpoint，如果存在則自動 Load 並從中斷的 Epoch/Step 繼續，而不是從頭開始。"
數據集與 Checkpoint 的「寫入集體自殺」防止機制（NFS/Ceph 殺手）
學生的 Checkpoint 通常極大（幾十 GB 到上百 GB）。如果一個大模型 Job 有 32 個節點（256卡），在同一個 Epoch 結束時同時調用 torch.save，會瞬間產生幾個 TB 的突發寫入流量。普通儲存會直接崩潰，導致整個集群其他人的 Job 全部 I/O Timeout。
DataLoader 沒有恢復狀態，續訓時又是從頭開始讀數據（或者是隨機洗牌），導致模型在前半段數據上過擬合，後半段數據沒看到。

Infra 必須提供： 推薦在 Template 中引入 WebDataset 的 Resumable 機制，或者使用 Hugging Face Trainer 的 data_seed 配合 ignore_data_skip=False，確保 Resume 時數據流也能精確對齊中斷的那個 Step

Infra 必須提供：

Rank 0 寫入限制： 框架必須強制只有 rank == 0 的進程負責寫入權重（如果是 ZeRO-3，則需使用 DeepSpeed 自帶的分片儲存，但要限制併發寫入的線程數）。

快取緩衝（Staging）： Checkpoint 先高速寫入到本地節點的 NVMe SSD（/scratch 盤），寫完後由後台異步複製到共享存儲。
唉 只要固定跑幾個epoch的時候存一次 然後重新跑得時候自動加載checkpoint就好 不用很精確某一次計算節點崩潰的時候要捕獲事件然後要存 不至於
3. dataset快取
 "千萬不要直接從網絡共享存儲（如普通的 NFS）上直接讀取包含數百萬張小圖片或無數文本小文件的數據集。這會造成嚴重的隨機 I/O 瓶頸" 
4. 跑job gpu節點排隊 整個中心的節點監控 看log等等
 "引入 PyTorch Elastic (torchrun)。確保代碼在啟動時會自動檢查是否存在最新的 Checkpoint，如果存在則自動 Load 並從中斷的 Epoch/Step 繼續，而不是從頭開始"
5. 編譯問題 通過 存cache
6. wandb 實驗狀況監控 結果輸出 中心化實驗追蹤
 "把 Hyperparameters、Loss 曲線、Gradient Norm、GPU 溫度、顯存佔用全部即時同步到雲端。這樣你躺在床上用手機都能看到實驗有沒有 NaN。"
7. evaluate 推理
8. 多卡互連跑 分散式訓練與網絡拓撲
 "NCCL 環境變量配置：
 H100/H200 節點內部通常是 NVLink，節點之間是 InfiniBand (IB) 或 RoCE 網絡。如果配置不對，PyTorch 會走慢速的 PCIe 甚至 TCP 網絡，速度直接掉 10 倍。

做法： 預先封裝好 NCCL 的 Debug 環境變量（如 export NCCL_DEBUG=INFO），並在 Infra 腳本中鎖定正確的網絡接口（如 NCCL_IB_DISABLE=0 確保啟用 IB）平行策略框架選型：
不要自己去手寫複雜的 mp.spawn。

做法： 根據模型大小，直接基於 Hugging Face Accelerate, DeepSpeed 或是 Megatron-LM 搭建你的實驗 Template。讓團隊成員只需要改寫單卡模型邏輯，Infra 層面用設定檔（Config）一鍵切換 DDP、ZeRO-2 或 ZeRO-3。"

「預標記與預處理」流水線化 把數據預處理做成獨立的 Offline Job 或是讓data argumentation讓gpu做
"文字類：提前把整個 Dataset 跑完 Tokenization，並用 Numpy Memmap 或是 Hugging Face 的 Arrow 格式存成大二進制文件。

訓練時，DataLoader 只需要做純粹的「記憶體映射（Memory-map）」，讀取速度是奈秒級別的，完全不佔用 CPU 算力"

10. "多租戶算力牆與心跳檢測（Zombie Job / Ghost Job 清理）
痛點： 在國家算力中心，有時候你的訓練腳本因為 CUDA Illegal Memory Access 崩潰了，但 PyTorch 进程沒死透，卡在後台變成「殭屍進程」，繼續霸佔著 H100 的顯存，導致下一個 Job 掛掉或者別人用不了。

解法： 寫一個 Slurm 的 epilog 腳本（或者學生自己寫一個清理工具）。在每個 Job 結束時，強行執行 pkill -u $USER -f python 或者是調用 nvidia-smi 檢查是否有殘留顯存，確保退出的時候機器是絕對乾淨的"

11. "模組功能,推薦現成工具,為什麼選它？（避坑理由）
分散式訓練/平行策略,Hugging Face Accelerate或 DeepSpeed,核心代碼只需要寫單卡邏輯。多卡 FSDP、ZeRO-3、張量平行（TP）全部透過一個 .yaml 設定檔一鍵切換。千萬別自己寫 torch.distributed。
實驗管理 / Hyperparameters,Hydra (Meta 開源),頂會論文標配。支援動態組合 Config、多實驗 Sweep（自動網格搜索）。你可以直接在命令行改參數，它會自動幫你建好對應的資料夾結構。
大數據集載入,WebDataset 或 Hugging Face Datasets,它們天生將海量小文件打包成 .tar 流式傳輸（Streaming），完美解決你最擔心的 NFS 隨機 I/O 崩潰問題。
集群一鍵提交與監控,Submitit (Meta 開源),專門為 Slurm 寫的 Python 封裝。讓學生可以在本地的 Jupyter Notebook 或是 Python 腳本裡，直接用對象調用的方式把函數「推」到 2000 卡集群上跑，自動收回 Log。"
Submitit 對於個人跑小實驗很爽，但在 2,000 卡的多租戶（Multi-tenant）環境下是運維災難。它會自動在後台生成大量隱藏的 .pkl 文件來序列化 Python 函數，一旦學生改動了本地代碼結構，恢復（Resume）時常會發生反序列化失敗（PickleError），且極難排查。

修正： 鼓勵學生撰寫標準的 .sbatch 腳本，Infra 層面提供一鍵生成的 CLI 工具（例如基於 Typer 或 Argparse 自研的提交腳本），而不是讓他們把整個 Python 環境序列化推上去

12. infra 修理 












1. ok 先做這個
2. 這個 你覺得怎麼辦比較好 我是傾向於我自己手動管理跟追蹤進程跑得怎樣 log那些
3. 這個可以做 但不是最優先
4. ok
5. ok 
6. ok
7. 這個很重要 你覺得怎麼做比較好
8. 我其實就在想了 我們既然都在邊訓練邊有那個把checkpoint存回去的機制 還有現在也有stdout stderr會進到log檔案 那為什麼不能邊訓練也一起存log資料？登入節點有網路 還是可以上傳wandb吧
9. 這個你覺得怎麼辦
10. 這個挺重要的 要顧到的邊界情況跟可能出的問題太多 要小心仔細 可能要討論一下



 第 7 點：結果輸出 / 收集                                                                                                                            
                                                                                                                                                     
 你的想法是對的 — 既然已經有 checkpoint 存回去的機制，metrics 也應該一起存。我建議分兩層：                                                           
                                                                                                                                                     
 Layer 1: train.py 每次 eval 後存 metrics                                                                                                            
 - 每個 epoch 結束後，把 metrics dict 寫到 runs/{exp_name}/metrics.jsonl（append mode，一行一個 JSON）                                               
 - 訓練結束時不需要額外動作，檔案已經完整                                                                                                            
 - 這樣即使 job 被 kill，已存的 epochs 的 metrics 都在                                                                                               
                                                                                                                                                     
 Layer 2: 登入節點上跑 collect_results.py                                                                                                            
 - 掃描所有 runs/*/metrics.jsonl                                                                                                                     
 - 匯出成一個 CSV/JSON summary table（exp_name, status, last_epoch, success_rate, ...）                                                              
 - 可以隨時跑，不需要等訓練結束                                                                                                                      
                                                                                                                                                     
 這樣你不用 grep log 檔，也不用 sync wandb 才能看到數字。metrics.jsonl 是結構化的，可以直接 pandas 讀。
 第 8 點：WandB 同步    
    
 你說得對 — 計算節點把 checkpoint 存回 /home（WekaFS），登入節點看得到。那 metrics 也能同樣存回去，登入節點有網路就能上傳。 
    
 但目前計算節點的 wandb 是 offline mode，wandb-osh 的機制是：   
 1. 計算節點寫 trigger file 到共享存儲 
 2. 登入節點跑 wandb sync daemon 消化 trigger → 上傳雲端   
    
 問題是 daemon 沒跑。兩個選擇：   
    
 選項 A: 啟動 wandb sync daemon（簡單）
 - 在登入節點跑 nohup wandb sync --sync-all wandb/ &  
 - 搭配第 7 點的 metrics.jsonl，等於雙保險  
    
 選項 B: 自己寫上傳腳本（更可控） 
 - train.py 存 metrics.jsonl 到 runs/（WekaFS 共享）  
 - 登入節點跑一個輕量 daemon，讀 metrics.jsonl，POST 到 wandb API（或直接用 wandb online mode 從登入節點 log）    
 - 好處是不依賴 wandb-osh，完全自己控制
    
 我傾向 A + 第 7 點的 metrics.jsonl。wandb-osh 已經有了，只需要啟動 daemon。metrics.jsonl 當備份和離線分析用。另外要改 train.py 的 wandb   
 entity/project（目前還是原作者的 wang-kevin3290-princeton-university）。 
    
 你覺得？還有 wandb entity 要改成什麼？env_setup.md 裡提到 sungwayne99999。    
    
 ────────────────────────────────────────────────────────────────────────────────   
    
 第 9 點：Snapshot daemon    
    
 現在的狀況： 
 - snapshot.py --loop 600 跑在 SSH session 裡（PID 455577）
 - 沒有 tmux/screen 保護，SSH 斷線就死 
 - monitor.sh 不會警告 snapshot 過期   
    
 建議：  
 1. 用 nohup 啟動，不受 SSH 斷線影響   
 2. monitor.sh 開頭加 snapshot age check — 如果 latest.json 超過 20 分鐘沒更新，印警告   
 3. 或者更簡單：不跑 daemon，monitor.sh 先跑一次 snapshot.py（單次）再顯示。這樣永遠是新資料，但每次查要等 2-3 秒。    
    
 你偏好哪種？ 
    
 ────────────────────────────────────────────────────────────────────────────────   
    
 第 10 點：Resume 機制邊界情況    
    
 我剛改了 train.py 處理了以下情況：    
    
 已修復：
 1. Corrupt checkpoint — load_params 改為 try/except 返回 None，不 crash  
 2. Auto-resume 找到 corrupt 檔 — find_latest_checkpoint 從新到舊試，跳過太小的檔（<100 bytes）    
 3. 全部 checkpoint 都 corrupt — 嘗試所有 checkpoint，都失敗就從頭開始    
 4. Atomic write — save_params_async 先寫 /tmp/.tmp 再 rename（local atomic），再 copy 到 WekaFS   
    
 還需要討論的：    
    
 5. cleanup_old_checkpoints 的 race condition — save_params_async 是 threading，如果 epoch N 的 thread 還沒寫完，epoch N+1 就呼叫
    cleanup_old_checkpoints(keep=3)，理論上 N 的檔案還在（因為 N 是倒數第 2 新），不會被刪。但如果連續多個 epoch 都在存（前 5 個 epoch
    每個都存），thread 積壓可能導致 cleanup 刪到正在寫的檔。不過實際上 keep=3 且前 5 epoch後改為每 10 epoch 存一次，積壓機率很低。    
 6. staging 失敗 — 如果 /tmp 滿了或不可寫，_save() 會在 thread 裡 crash，主程式不知道。可以加 try/except 在 thread 裡，失敗時 fallback 直接寫   
    WekaFS，或者 print warning。  
 7. 舊格式 checkpoint（bare tuple） — runs/_old/ 裡的舊 checkpoint 沒有 args.pkl，resume 時只會跳 warning 不 block。但這些舊目錄名不匹配   
    exp_name，auto-resume 找不到，只有手動指定才會碰到。基本上不是問題。  
