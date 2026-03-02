# Phân tích notebook `Matryoshka_Adaptor.ipynb` theo từng cell

## Cell 1
1) **Mục đích:** Gắn badge mở notebook trực tiếp trên Google Colab.  
2) **Giải thích code:** Đây là HTML `<a><img></a>` trỏ đến URL Colab của repo. Không có Python runtime logic.  
3) **Biến quan trọng:** Không có.  
4) **Ảnh hưởng cell sau:** Không ảnh hưởng execution.  
5) **Giả định ẩn:** Người dùng chạy notebook qua Colab hoặc môi trường có internet.  
6) **Refactor:** Không cần.

## Cell 2
1) **Mục đích:** Mô tả luồng chính: train unsupervised Matryoshka-Adaptor và đánh giá retrieval.  
2) **Giải thích code:** Markdown mô tả train/eval dataset, có ghi chú có thể đổi `tr_dataset_name`.  
3) **Biến quan trọng:** Không có biến runtime.  
4) **Ảnh hưởng cell sau:** Định hướng người đọc về các khối train/eval.  
5) **Giả định ẩn:** Text hơi lệch với config hiện tại (nói SciFact train, nhưng config đang đặt MSMARCO).  
6) **Refactor:** Nên cập nhật mô tả để khớp hyperparameter thực tế.

## Cell 3
1) **Mục đích:** Cài dependency.  
2) **Giải thích code:** `pip install sentence-transformers beir mteb faiss-cpu`. Các package này phục vụ encode, tải BEIR dataset, benchmark MTEB, KNN FAISS.  
3) **Biến quan trọng:** Không tạo biến Python, nhưng tạo môi trường package.  
4) **Ảnh hưởng cell sau:** Bắt buộc cho import ở cell 4 và toàn bộ pipeline train/eval.  
5) **Giả định ẩn:** Có internet, có quyền cài package. `faiss-cpu` có thể chậm hơn GPU FAISS.  
6) **Refactor:** Có thể pin version để reproducible hơn.

## Cell 4
1) **Mục đích:** Import toàn bộ thư viện cần dùng.  
2) **Giải thích code:** Import BEIR loader/util, SentenceTransformer, PyTorch, FAISS, MTEB, DataLoader/Dataset (nhưng chưa dùng), matplotlib, typing.  
3) **Biến quan trọng:** Namespace của module (`torch`, `faiss`, `MTEB`, ...).  
4) **Ảnh hưởng cell sau:** Tất cả function/class ở các cell sau phụ thuộc các import này.  
5) **Giả định ẩn:** Các package ở cell 3 đã cài thành công.  
6) **Refactor:** Có thể bỏ import chưa dùng (`LoggingHandler`, `DataLoader`, `Dataset`, `os` nếu không dùng).

## Cell 5
1) **Mục đích:** Thiết lập random seed cho tái lập tương đối.  
2) **Giải thích code:** Tạo `random_seed=0`; tạo RNG riêng `python_random`; gọi `torch.manual_seed`.  
3) **Biến quan trọng:** `random_seed`, `python_random`.  
4) **Ảnh hưởng cell sau:** Sampling corpus (`python_random.sample`) và sinh cặp index sanity check reproducible hơn.  
5) **Giả định ẩn:** Chưa set đầy đủ deterministic cho CUDA/cuDNN nên vẫn có nhiễu giữa lần chạy.  
6) **Refactor:** Có thể thêm `np.random.seed`, `torch.cuda.manual_seed_all`, và cờ deterministic.

## Cell 6
1) **Mục đích:** Header phân vùng Hyperparameters.  
2) **Giải thích code:** Markdown.  
3) **Biến quan trọng:** Không có.  
4) **Ảnh hưởng cell sau:** Không trực tiếp.  
5) **Giả định ẩn:** Không.  
6) **Refactor:** Không cần.

## Cell 7
1) **Mục đích:** Khai báo cấu hình train/eval và device.  
2) **Giải thích code:** Đặt base model (`all-MiniLM-L6-v2`), `embedding_dim=384`, list prefix `m_values=[64,128,256,384]`, hidden dim adaptor, dataset train/eval, corpus cap (`MAX_DOCS`), debug switch, batch/lr/patience/max_iter/logging/grad-clip. Định nghĩa trọng số loss Eq.4 (`LAMBDA_*`) và profile tuning (`TRAIN_LAMBDA_*`). Chọn `device = cuda if available else cpu`, in config.  
3) **Biến quan trọng:** `base_encoder`, `m_values`, `k`, `batch_size`, `optimizer_fn`, `patience`, `max_tr_iterations`, `TRAIN_LAMBDA_*`, `device`, `max_num_tr_datapoints`.  
4) **Ảnh hưởng cell sau:** Điều khiển gần như toàn bộ data loading, loss, training loop, evaluation truncation.  
5) **Giả định ẩn:** `embedding_dim` phải khớp dim output model base (384). `k` phải nhỏ hơn số mẫu thực tế.  
6) **Refactor:** Nên gom vào dataclass/config object để giảm biến global rời rạc.

## Cell 8
1) **Mục đích:** Header phần model adaptor.  
2) **Giải thích code:** Markdown.  
3) **Biến quan trọng:** Không có.  
4) **Ảnh hưởng cell sau:** Không trực tiếp.  
5) **Giả định ẩn:** Không.  
6) **Refactor:** Không cần.

## Cell 9
1) **Mục đích:** Định nghĩa kiến trúc Matryoshka Adaptor.  
2) **Giải thích code:** `SwiGLU`: 2 linear song song, một nhánh qua SiLU rồi nhân phần tử với nhánh còn lại. `MatryoshkaAdaptor`: `SwiGLU(384->1024)` + `Linear(1024->384)` + residual `out+inputs`. `forward` hỗ trợ cả tensor thuần lẫn dict kiểu SentenceTransformer (`{"sentence_embedding":...}`).  
3) **Biến quan trọng:** Class `SwiGLU`, `MatryoshkaAdaptor`; các layer `linear_1`, `linear_2`, `layer1`, `layer2`.  
4) **Ảnh hưởng cell sau:** Là module được train ở cell 22 và gắn vào pipeline eval ở cell 31+.  
5) **Giả định ẩn:** Input phải có dim cuối bằng `embedding_dim`; nếu dict thì key `sentence_embedding` tồn tại.  
6) **Refactor:** Có thể thêm type-check rõ hơn và assert shape để lỗi sớm.

## Cell 10
1) **Mục đích:** Header phần chuẩn bị dữ liệu.  
2) **Giải thích code:** Markdown.  
3) **Biến quan trọng:** Không có.  
4) **Ảnh hưởng cell sau:** Không trực tiếp.  
5) **Giả định ẩn:** Không.  
6) **Refactor:** Không cần.

## Cell 11
1) **Mục đích:** Tải BEIR train corpus và encode thành embedding cho train unsupervised.  
2) **Giải thích code:** Download zip BEIR theo `dataset_name`, load split `train`; lấy `corpus_keys`, nếu vượt cap thì sample bằng `python_random`; load SentenceTransformer base, `.to(device)`; encode toàn bộ text (`convert_to_numpy=False`) rồi `torch.stack`; giải phóng `corpus` và `model`, gọi `torch.cuda.empty_cache`; trả tensor embeddings.  
3) **Biến quan trọng:** Trong hàm: `corpus`, `corpus_keys`, `model`, `corpus_embeddings`.  
4) **Ảnh hưởng cell sau:** Tạo `corpus_embeddings` nền cho KNN, loss, train adaptor.  
5) **Giả định ẩn:** `model.encode(..., convert_to_numpy=False)` trả iterable tensor tương thích `torch.stack`; corpus text đủ sạch; đủ RAM/VRAM cho encode khối lớn.  
6) **Refactor:** Nên encode theo chunk streaming để tránh giữ toàn bộ list trong RAM.

## Cell 12
1) **Mục đích:** Tính KNN theo cosine dùng FAISS để phục vụ `loss_knn`.  
2) **Giải thích code:** L2-normalize embeddings => cosine = inner product; chuyển numpy float32; build `IndexFlatIP`; search `k+1` rồi loại self-index; đảm bảo mỗi hàng còn đủ `k` neighbor, nếu thiếu thì raise error; trả về ma trận chỉ số `(N,k)`.  
3) **Biến quan trọng:** `index`, `indices`, `knn_indices`.  
4) **Ảnh hưởng cell sau:** `knn_indices` được dùng trực tiếp trong training loop để lấy hàng xóm cho mỗi sample.  
5) **Giả định ẩn:** N đủ lớn để có ít nhất `k` neighbors ngoài chính nó; embeddings không NaN.  
6) **Refactor:** Có thể vectorize khâu lọc self để giảm vòng lặp Python.

## Cell 13
1) **Mục đích:** Header phần loss.  
2) **Giải thích code:** Markdown.  
3) **Biến quan trọng:** Không có.  
4) **Ảnh hưởng cell sau:** Không trực tiếp.  
5) **Giả định ẩn:** Không.  
6) **Refactor:** Không cần.

## Cell 14
1) **Mục đích:** Định nghĩa `L_rec` (reconstruction-style).  
2) **Giải thích code:** Normalize embedding đã transform; tính `mean(abs(original - transformed_norm))` trên toàn batch và toàn chiều.  
3) **Biến quan trọng:** `transformed_embeddings_normalized`.  
4) **Ảnh hưởng cell sau:** Một thành phần của tổng loss Eq.4 trong training loop.  
5) **Giả định ẩn:** `embeddings` đầu vào đã chuẩn hóa (thực tế training_loop cũng normalize trước).  
6) **Refactor:** Biến `batch_size` được tính nhưng không dùng -> nên bỏ.

## Cell 15
1) **Mục đích:** Định nghĩa `L_pair` giữ cấu trúc similarity pairwise qua nhiều prefix m.  
2) **Giải thích code:** Tính ma trận similarity gốc `sim_original = E E^T`; tạo mask tam giác trên để bỏ diagonal/self-similarity; với mỗi `m` trong `m_values`, cắt prefix `:m`, normalize lại, tính similarity mới, lấy MAE giữa 2 similarity matrix trên vùng `upper_mask`; trung bình qua các m.  
3) **Biến quan trọng:** `sim_original`, `upper_mask`, `per_m_losses`.  
4) **Ảnh hưởng cell sau:** Thành phần loss giúp adaptor giữ quan hệ cặp điểm khi truncate.  
5) **Giả định ẩn:** `m_values` là biến global hợp lệ và `m<=D`.  
6) **Refactor:** Tính similarity full NxN có thể nặng O(N²) theo batch lớn; có thể thay sampled pairs.

## Cell 16
1) **Mục đích:** Định nghĩa `L_knn` bảo toàn độ tương tự với k láng giềng gần nhất.  
2) **Giải thích code:** Normalize embeddings và knn embeddings; tính cosine gốc giữa mỗi điểm và k neighbors. Với mỗi `m`: cắt prefix cho transformed main + transformed knn, normalize, tính cosine prefix; lấy MAE với cosine gốc; trung bình qua m.  
3) **Biến quan trọng:** `embeddings_similarities`, `c_transformed_embeddings_similarities`, `per_m_losses`.  
4) **Ảnh hưởng cell sau:** Thành phần loss chính để duy trì local neighborhood khi giảm chiều prefix.  
5) **Giả định ẩn:** Tensor shapes: `(B,D)`, `(B,k,D)`, `(B,k,D)` phải đồng nhất; `k` đúng với KNN precompute.  
6) **Refactor:** Có thể giảm memory bằng tính theo block khi `B*k*D` lớn.

## Cell 17
1) **Mục đích:** Header phần training.  
2) **Giải thích code:** Markdown.  
3) **Biến quan trọng:** Không có.  
4) **Ảnh hưởng cell sau:** Không trực tiếp.  
5) **Giả định ẩn:** Không.  
6) **Refactor:** Không cần.

## Cell 18
1) **Mục đích:** Vòng lặp huấn luyện adaptor với objective tổng hợp Eq.4.  
2) **Giải thích code:** 
- Chuẩn bị device: lấy `model_device` từ tham số adaptor; move+normalize `embeddings`; move `knn_indices`.  
- Epoch loop tới `max_tr_iterations`; random permutation mỗi epoch.  
- Mỗi batch: lấy `batch_embeddings`, `batch_knn_indices`, assert số cột KNN = `k`, gather `batch_knn_embeddings`.  
- Forward: adaptor cho batch chính và batch neighbor (flatten rồi reshape về `(B,k,D)`).  
- Loss: `l_rec`, `l_pair`, `l_knn`; tổng `loss = λ_rec*l_rec + λ_pair*l_pair + λ_knn*l_knn`.  
- Ổn định train: check NaN/Inf, backward, gradient clipping (`clip_grad_norm_`), optimizer.step().  
- Logging định kỳ theo `log_every`; cuối epoch tính trung bình từng loss thành phần; early stopping theo `patience` nếu `epoch_loss` không cải thiện.  
3) **Biến quan trọng:** `best_loss`, `epochs_without_improvement`, `total_iterations`, `loss/l_rec/l_pair/l_knn`, `indices`, `batch_knn_embeddings_processed`.  
4) **Ảnh hưởng cell sau:** Trực tiếp cập nhật trọng số `matryoshka_adaptor` để dùng trong sanity check và eval MTEB.  
5) **Giả định ẩn:** Optimizer đã bind đúng params; `knn_indices` khớp `embeddings`; loss functions phụ thuộc global `m_values`; việc normalize đầu vào không làm mất thông tin cần học.  
6) **Refactor:** Nên trả về history dict (loss curve) để tiện plot/so sánh; hiện tại chỉ print.

## Cell 19
1) **Mục đích:** Chạy data prep thực tế cho train.  
2) **Giải thích code:** Gọi `load_and_preprocess_data(tr_dataset_name, base_encoder)` tạo `corpus_embeddings`; normalize L2 lần nữa; in shape.  
3) **Biến quan trọng:** `corpus_embeddings`.  
4) **Ảnh hưởng cell sau:** Input chính cho KNN (cell 20) và train (cell 22).  
5) **Giả định ẩn:** Download dataset và encode thành công, shape `(N,384)`.  
6) **Refactor:** Có thể cache embeddings ra disk để tránh encode lại mỗi lần chạy.

## Cell 20
1) **Mục đích:** Tính và đóng gói KNN indices.  
2) **Giải thích code:** Gọi `compute_knn_cosine`; in shape; convert numpy -> torch tensor `corpus_embeddings_knn_indices_torch`.  
3) **Biến quan trọng:** `corpus_embeddings_knn_indices`, `corpus_embeddings_knn_indices_torch`.  
4) **Ảnh hưởng cell sau:** Dùng trực tiếp trong `training_loop`.  
5) **Giả định ẩn:** KNN trả đúng `(N,k)` và dtype index tương thích indexing PyTorch.  
6) **Refactor:** Nên ép dtype `long` rõ ràng để tránh rủi ro (`torch.from_numpy(...).long()`).

## Cell 21
1) **Mục đích:** Khởi tạo model adaptor và optimizer.  
2) **Giải thích code:** Tạo `MatryoshkaAdaptor(embedding_dim, ma_hidden_dim).to(device)`; optimizer lấy từ `optimizer_fn` (Adam lr=1e-3).  
3) **Biến quan trọng:** `matryoshka_adaptor`, `optimizer`.  
4) **Ảnh hưởng cell sau:** Là đối tượng được train (cell 22) và eval (31+).  
5) **Giả định ẩn:** `embedding_dim` đúng dim đầu vào embeddings.  
6) **Refactor:** Có thể in số params trainable để debug.

## Cell 22
1) **Mục đích:** Huấn luyện adaptor.  
2) **Giải thích code:** Gọi `training_loop(...)` với dữ liệu embeddings, KNN indices, optimizer, batch/iter/patience, trọng số loss tuned (`TRAIN_LAMBDA_*`), log/grad clip.  
3) **Biến quan trọng:** Cập nhật nội bộ weight của `matryoshka_adaptor`.  
4) **Ảnh hưởng cell sau:** Quyết định chất lượng sanity checks và benchmark retrieval.  
5) **Giả định ẩn:** Đủ compute thời gian/VRAM; dữ liệu train đã normalize + KNN hợp lệ.  
6) **Refactor:** Nên lưu checkpoint tốt nhất theo `best_loss` thay vì chỉ giữ bản cuối.

## Cell 23
1) **Mục đích:** Header phần save/load model.  
2) **Giải thích code:** Markdown.  
3) **Biến quan trọng:** Không có.  
4) **Ảnh hưởng cell sau:** Không trực tiếp.  
5) **Giả định ẩn:** Không.  
6) **Refactor:** Không cần.

## Cell 24
1) **Mục đích:** Mẫu code save/load state_dict (đang comment).  
2) **Giải thích code:** Nếu bỏ comment: `torch.save(state_dict)` -> khởi tạo adaptor mới -> `load_state_dict`.  
3) **Biến quan trọng:** `file_path` (khi bật).  
4) **Ảnh hưởng cell sau:** Cho phép tái sử dụng adaptor đã train mà không train lại.  
5) **Giả định ẩn:** Kiến trúc model lúc load phải trùng với lúc save.  
6) **Refactor:** Có thể đóng gói thành hàm `save_adaptor`/`load_adaptor`.

## Cell 25
1) **Mục đích:** Cell trống (placeholder).  
2) **Giải thích code:** Không có nội dung.  
3) **Biến quan trọng:** Không có.  
4) **Ảnh hưởng cell sau:** Không.  
5) **Giả định ẩn:** Không.  
6) **Refactor:** Nên xóa để notebook gọn hơn.

## Cell 26
1) **Mục đích:** Header sanity check Matryoshka property.  
2) **Giải thích code:** Markdown mô tả mục tiêu kiểm tra unsupervised.  
3) **Biến quan trọng:** Không có.  
4) **Ảnh hưởng cell sau:** Định hướng 2 test ở cell 28-29.  
5) **Giả định ẩn:** Không.  
6) **Refactor:** Không cần.

## Cell 27
1) **Mục đích:** Chuẩn bị tensor gốc và tensor đã qua adaptor cho sanity checks.  
2) **Giải thích code:** `matryoshka_adaptor.eval()`, tạo `corpus_embeddings_orig` normalized CPU; `torch.no_grad()` để suy luận adaptor trên `corpus_embeddings_orig.to(device)`; normalize output rồi chuyển CPU; in shape.  
3) **Biến quan trọng:** `corpus_embeddings_orig`, `corpus_embeddings_adapted`.  
4) **Ảnh hưởng cell sau:** Là dữ liệu đầu vào cho overlap@10 và cosine-correlation test.  
5) **Giả định ẩn:** Model đã train xong; device chuyển qua lại CPU/GPU thành công.  
6) **Refactor:** Có thể tách thành hàm `build_eval_embeddings` tái sử dụng.

## Cell 28
1) **Mục đích:** Đánh giá bảo toàn lân cận bằng Neighbor Overlap@10.  
2) **Giải thích code:** Định nghĩa helper `faiss_topk_no_self`; lấy sample query IDs tối đa 200; `gold_neighbors` lấy từ full-dim original; với mỗi `m`: truncate original và adapted về `:m`, normalize lại, truy hồi top-10; tính tỉ lệ giao nhau trung bình với gold. In bảng `baseline_truncate` vs `adaptor`.  
3) **Biến quan trọng:** `sample_ids`, `gold_neighbors`, `base_overlap`, `adapt_overlap`.  
4) **Ảnh hưởng cell sau:** Không dùng trực tiếp cho train/eval MTEB nhưng cung cấp sanity evidence nhanh.  
5) **Giả định ẩn:** Sample size 200 đủ đại diện; cosine top-k bằng IndexFlatIP sau normalize.  
6) **Refactor:** Có thể thêm độ lệch chuẩn/CI thay vì chỉ mean.

## Cell 29
1) **Mục đích:** Đánh giá bảo toàn cấu trúc cosine toàn cục bằng Pearson + MSE.  
2) **Giải thích code:** Sinh ngẫu nhiên các cặp `(i,j)` khác nhau (tối đa 4000 hoặc N*8); `cos_full` từ original full-dim. Với mỗi `m`, tính `cos_base` (truncate baseline) và `cos_adapt`; sau đó tính `pearson` với `cos_full` và `mse` sai khác; in bảng; in thống kê norm original/adapted.  
3) **Biến quan trọng:** `pair_i/pair_j`, `cos_full`, `pearson_base/adapt`, `mse_base/adapt`.  
4) **Ảnh hưởng cell sau:** Cũng là sanity diagnostic độc lập, không tác động flow chính.  
5) **Giả định ẩn:** Cặp random đủ phủ phân phối similarity; `np.corrcoef` ổn định khi variance đủ lớn.  
6) **Refactor:** Có thể vector hóa sinh cặp không-trùng bằng phương pháp trực tiếp thay while-loop.

## Cell 30
1) **Mục đích:** Header phần evaluation retrieval chính thức.  
2) **Giải thích code:** Markdown.  
3) **Biến quan trọng:** Không có.  
4) **Ảnh hưởng cell sau:** Không trực tiếp.  
5) **Giả định ẩn:** Không.  
6) **Refactor:** Không cần.

## Cell 31
1) **Mục đích:** Tạo wrapper `SentenceTransformer` để chèn adaptor đúng pipeline eval Matryoshka.  
2) **Giải thích code:** 
- `__init__`: khởi tạo base model, chọn `adaptor_device`, move adaptor và `.eval()`.  
- `encode`: lưu `truncate_dim` hiện tại, ép base encode trả tensor full-dim (`self.truncate_dim=None`, `convert_to_tensor=True`), normalize base, chạy adaptor, normalize lại, rồi mới truncate `:m` nếu có. Sau đó trả về theo format người gọi yêu cầu (tensor/numpy/list). `finally` khôi phục `truncate_dim`.  
- `encode_queries` và `encode_corpus`: alias tương thích API MTEB/BEIR; nếu corpus là list dict thì ghép `title + text`.  
3) **Biến quan trọng:** `SentenceTransformerwMatryoskaAdaptor`, `self.adaptor_device`, `m_value`.  
4) **Ảnh hưởng cell sau:** Là lõi pipeline evaluation ở cell 34-35, đảm bảo thứ tự **encode full -> adaptor -> truncate**.  
5) **Giả định ẩn:** `truncate_dim` tồn tại/được MTEB dùng; adaptor output cùng dim base.  
6) **Refactor:** Tên class có typo “Matryoska” (thiếu h) -> nên sửa đồng nhất để dễ bảo trì.

## Cell 32
1) **Mục đích:** Header config eval.  
2) **Giải thích code:** Markdown.  
3) **Biến quan trọng:** Không có.  
4) **Ảnh hưởng cell sau:** Không trực tiếp.  
5) **Giả định ẩn:** Không.  
6) **Refactor:** Không cần.

## Cell 33
1) **Mục đích:** Cấu hình benchmark retrieval.  
2) **Giải thích code:** Đặt dataset eval (`["scifact"]`), `EVAL_M_VALUES = m_values`, metric key `ndcg_at_10`, biến tùy chọn cap query (chưa dùng), in config.  
3) **Biến quan trọng:** `EVAL_DATASETS`, `EVAL_M_VALUES`, `METRIC`, `MAX_EVAL_QUERIES`.  
4) **Ảnh hưởng cell sau:** Điều khiển loop đánh giá ở cell 35.  
5) **Giả định ẩn:** Metric key khớp đúng key output MTEB của task retrieval.  
6) **Refactor:** Nếu dùng `MAX_EVAL_QUERIES`, cần tích hợp vào hàm eval để có tác dụng thực.

## Cell 34
1) **Mục đích:** Định nghĩa utility/hàm chạy MTEB baseline-vs-adaptor.  
2) **Giải thích code:** Import `Path/json/pandas`; map tên dataset -> task name MTEB; `_extract_metric` bóc `eval_output[0].scores["test"][0][metric_key]`. `run_mteb_eval_milestone3`: với mỗi dataset tạo task+benchmark, khởi tạo baseline model và adaptor wrapper; với mỗi m: set `truncate_dim=m` cho từng model, chạy `benchmark.run`, lấy metric, lưu baseline/adaptor/delta và print; trả dict `results`.  
3) **Biến quan trọng:** `MTEB_TASK_NAME_MAP`, `results`, `dataset_results`.  
4) **Ảnh hưởng cell sau:** Cell 35 gọi hàm này để tạo kết quả chính thức.  
5) **Giả định ẩn:** Cấu trúc output MTEB ổn định theo indexing `[0]...`; chạy benchmark lặp lại nhiều lần (mỗi m) chấp nhận được thời gian.  
6) **Refactor:** Có thể cache embedding/query để tránh chạy lại full benchmark cho mỗi `m`.

## Cell 35
1) **Mục đích:** Thực thi evaluation và validation nhanh.  
2) **Giải thích code:** Gọi `run_mteb_eval_milestone3(...)` với model base + adaptor đã train; sau đó duyệt từng dataset, kiểm tra có ít nhất một `delta` khác 0 (`abs(d)>1e-12`) và in kết quả.  
3) **Biến quan trọng:** `results`, `deltas`, `has_difference`.  
4) **Ảnh hưởng cell sau:** `results` là đầu vào cho bảng (cell 36), biểu đồ (37), và lưu JSON (38).  
5) **Giả định ẩn:** `matryoshka_adaptor` đã tồn tại/trained; không ở chế độ train ảnh hưởng dropout (ở đây model adaptor đã `.eval()` trong wrapper).  
6) **Refactor:** Có thể thêm assert cứng nếu `has_difference=False` để fail-fast khi pipeline adaptor không tác dụng.

## Cell 36
1) **Mục đích:** Trình bày kết quả thành bảng DataFrame.  
2) **Giải thích code:** Tạo `result_tables` dict; với mỗi dataset đóng gói cột `m`, `baseline`, `adaptor`, `delta`; `display(table)`.  
3) **Biến quan trọng:** `result_tables`, `table`.  
4) **Ảnh hưởng cell sau:** Cell 37 dùng `result_tables` để vẽ đồ thị.  
5) **Giả định ẩn:** Chạy trong notebook có `display` (Jupyter/Colab).  
6) **Refactor:** Có thể format số thập phân để dễ đọc hơn.

## Cell 37
1) **Mục đích:** Vẽ biểu đồ baseline vs adaptor theo từng m.  
2) **Giải thích code:** Loop qua từng dataset table, `plt.plot` hai đường, thêm title/label/grid/legend rồi `show()`.  
3) **Biến quan trọng:** Không tạo biến mới ngoài `dataset`, `table` trong loop.  
4) **Ảnh hưởng cell sau:** Không ảnh hưởng logic; chỉ trực quan hóa.  
5) **Giả định ẩn:** Matplotlib backend hoạt động trong môi trường notebook.  
6) **Refactor:** Có thể lưu figure ra file để dùng ngoài notebook.

## Cell 38
1) **Mục đích:** Lưu kết quả eval để tái lập/so sánh run.  
2) **Giải thích code:** Tạo `results_path="results_mteb_eval.json"`; convert key `m` sang string để JSON serializable; `json.dump(..., indent=2)`; in đường dẫn tuyệt đối.  
3) **Biến quan trọng:** `results_path`, `serializable_results`.  
4) **Ảnh hưởng cell sau:** Không còn cell sau; tạo artifact đầu ra.  
5) **Giả định ẩn:** Có quyền ghi file tại working directory.  
6) **Refactor:** Có thể thêm timestamp/seed trong tên file để lưu nhiều run không ghi đè.

## Tổng kết logic train/eval theo yêu cầu
- **Training loss/optimizer/device:** Objective ở cell 18 là tổng có trọng số của `loss_rec` (cell 14), `loss_pair` (cell 15), `loss_knn` (cell 16) với các λ từ cell 7/22; optimizer là Adam lr=1e-3; device handling gồm move embeddings + knn indices sang cùng device model, normalize trước loss, gradient clipping để ổn định.  
- **Evaluation pipeline (encode/truncate/adaptor/metric):** Wrapper cell 31 ép quy trình đúng thứ tự `base full-dim encode -> normalize -> adaptor -> normalize -> truncate[:m]`; cell 34 so baseline truncation thuần với adaptor pipeline và lấy metric `ndcg_at_10`; cell 35-38 tổng hợp, hiển thị, plot, lưu JSON.
