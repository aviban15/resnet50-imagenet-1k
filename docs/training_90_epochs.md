2025-10-23 15:06:39,173 - INFO - === ResNet50 ImageNet-1K Training Started ===
2025-10-23 15:06:39,173 - INFO - Loading ImageNet dataloaders...
2025-10-23 15:06:41,925 - INFO - Train loader: 10010 batches
2025-10-23 15:06:41,925 - INFO - Validation loader: 391 batches
2025-10-23 15:06:41,925 - INFO - Loading ResNet model...
2025-10-23 15:06:42,198 - INFO - ResNet model loaded successfully
2025-10-23 15:06:42,330 - INFO - Using device = cuda
2025-10-23 15:06:42,330 - INFO - Training configuration: 90 epochs
2025-10-23 15:06:42,516 - INFO - Optimizer: SGD (
Parameter Group 0
    dampening: 0
    differentiable: False
    foreach: None
    fused: None
    initial_lr: 0.1
    lr: 0.1
    maximize: False
    momentum: 0.9
    nesterov: False
    weight_decay: 0.0001
)
2025-10-23 15:06:42,516 - INFO - Scheduler: <torch.optim.lr_scheduler.CosineAnnealingLR object at 0x763b6b8b5a90>
2025-10-23 15:06:42,516 - INFO - Mixed precision training enabled
2025-10-23 15:06:42,517 - INFO - Starting fresh training...
2025-10-23 15:06:42,517 - INFO - Starting training loop...
2025-10-23 15:06:42,517 - INFO - ============================================================
2025-10-23 15:06:42,517 - INFO - EPOCH: 1/90 | LR: 0.100000
2025-10-23 15:42:46,811 - INFO - Epoch 1 Summary:
2025-10-23 15:42:46,812 - INFO -   Train Loss: 5.0273, Train Accuracy: 6.92%
2025-10-23 15:42:46,812 - INFO -   Validation Loss: 4.4923, Validation Accuracy: 14.59%
2025-10-23 15:42:47,076 - INFO - Checkpoint saved
2025-10-23 15:42:47,211 - INFO - New best accuracy: 14.59% - Best model weights saved
2025-10-23 15:42:47,211 - INFO - Epoch 1 completed and logged.
2025-10-23 15:42:47,211 - INFO - ============================================================
2025-10-23 15:42:47,211 - INFO - EPOCH: 2/90 | LR: 0.099970
2025-10-23 16:18:47,408 - INFO - Epoch 2 Summary:
2025-10-23 16:18:47,409 - INFO -   Train Loss: 3.5367, Train Accuracy: 23.49%
2025-10-23 16:18:47,409 - INFO -   Validation Loss: 3.4010, Validation Accuracy: 29.57%
2025-10-23 16:18:48,228 - INFO - Checkpoint saved
2025-10-23 16:18:48,996 - INFO - New best accuracy: 29.57% - Best model weights saved
2025-10-23 16:18:48,997 - INFO - Epoch 2 completed and logged.
2025-10-23 16:18:48,997 - INFO - ============================================================
2025-10-23 16:18:48,997 - INFO - EPOCH: 3/90 | LR: 0.099878
2025-10-23 16:54:51,783 - INFO - Epoch 3 Summary:
2025-10-23 16:54:51,784 - INFO -   Train Loss: 3.1529, Train Accuracy: 31.58%
2025-10-23 16:54:51,784 - INFO -   Validation Loss: 2.9872, Validation Accuracy: 35.13%
2025-10-23 16:54:52,630 - INFO - Checkpoint saved
2025-10-23 16:54:53,413 - INFO - New best accuracy: 35.13% - Best model weights saved
2025-10-23 16:54:53,414 - INFO - Epoch 3 completed and logged.
2025-10-23 16:54:53,414 - INFO - ============================================================
2025-10-23 16:54:53,414 - INFO - EPOCH: 4/90 | LR: 0.099726
2025-10-23 17:30:58,480 - INFO - Epoch 4 Summary:
2025-10-23 17:30:58,481 - INFO -   Train Loss: 3.9049, Train Accuracy: 35.65%
2025-10-23 17:30:58,481 - INFO -   Validation Loss: 2.8424, Validation Accuracy: 37.72%
2025-10-23 17:30:59,297 - INFO - Checkpoint saved
2025-10-23 17:31:00,073 - INFO - New best accuracy: 37.72% - Best model weights saved
2025-10-23 17:31:00,073 - INFO - Epoch 4 completed and logged.
2025-10-23 17:31:00,073 - INFO - ============================================================
2025-10-23 17:31:00,073 - INFO - EPOCH: 5/90 | LR: 0.099514
2025-10-23 18:07:06,109 - INFO - Epoch 5 Summary:
2025-10-23 18:07:06,110 - INFO -   Train Loss: 2.8737, Train Accuracy: 38.07%
2025-10-23 18:07:06,110 - INFO -   Validation Loss: 2.7023, Validation Accuracy: 39.69%
2025-10-23 18:07:06,913 - INFO - Checkpoint saved
2025-10-23 18:07:07,695 - INFO - New best accuracy: 39.69% - Best model weights saved
2025-10-23 18:07:07,695 - INFO - Epoch 5 completed and logged.
2025-10-23 18:07:07,695 - INFO - ============================================================
2025-10-23 18:07:07,695 - INFO - EPOCH: 6/90 | LR: 0.099241
2025-10-23 18:43:14,467 - INFO - Epoch 6 Summary:
2025-10-23 18:43:14,467 - INFO -   Train Loss: 2.9288, Train Accuracy: 39.60%
2025-10-23 18:43:14,468 - INFO -   Validation Loss: 2.5813, Validation Accuracy: 42.08%
2025-10-23 18:43:15,275 - INFO - Checkpoint saved
2025-10-23 18:43:16,065 - INFO - New best accuracy: 42.08% - Best model weights saved
2025-10-23 18:43:16,065 - INFO - Epoch 6 completed and logged.
2025-10-23 18:43:16,065 - INFO - ============================================================
2025-10-23 18:43:16,065 - INFO - EPOCH: 7/90 | LR: 0.098908
2025-10-23 19:19:27,000 - INFO - Epoch 7 Summary:
2025-10-23 19:19:27,001 - INFO -   Train Loss: 2.6524, Train Accuracy: 40.67%
2025-10-23 19:19:27,001 - INFO -   Validation Loss: 2.4315, Validation Accuracy: 44.24%
2025-10-23 19:19:27,823 - INFO - Checkpoint saved
2025-10-23 19:19:28,600 - INFO - New best accuracy: 44.24% - Best model weights saved
2025-10-23 19:19:28,600 - INFO - Epoch 7 completed and logged.
2025-10-23 19:19:28,600 - INFO - ============================================================
2025-10-23 19:19:28,600 - INFO - EPOCH: 8/90 | LR: 0.098516
2025-10-23 19:55:38,045 - INFO - Epoch 8 Summary:
2025-10-23 19:55:38,049 - INFO -   Train Loss: 2.3465, Train Accuracy: 41.45%
2025-10-23 19:55:38,049 - INFO -   Validation Loss: 2.4088, Validation Accuracy: 45.00%
2025-10-23 19:55:38,893 - INFO - Checkpoint saved
2025-10-23 19:55:39,682 - INFO - New best accuracy: 45.00% - Best model weights saved
2025-10-23 19:55:39,682 - INFO - Epoch 8 completed and logged.
2025-10-23 19:55:39,682 - INFO - ============================================================
2025-10-23 19:55:39,682 - INFO - EPOCH: 9/90 | LR: 0.098065
2025-10-23 20:31:49,685 - INFO - Epoch 9 Summary:
2025-10-23 20:31:49,686 - INFO -   Train Loss: 3.4762, Train Accuracy: 42.07%
2025-10-23 20:31:49,686 - INFO -   Validation Loss: 2.7077, Validation Accuracy: 40.19%
2025-10-23 20:31:50,494 - INFO - Checkpoint saved
2025-10-23 20:31:50,494 - INFO - Epoch 9 completed and logged.
2025-10-23 20:31:50,494 - INFO - ============================================================
2025-10-23 20:31:50,494 - INFO - EPOCH: 10/90 | LR: 0.097555
2025-10-23 21:10:02,548 - INFO - Epoch 10 Summary:
2025-10-23 21:10:02,550 - INFO -   Train Loss: 3.4439, Train Accuracy: 42.55%
2025-10-23 21:10:02,550 - INFO -   Validation Loss: 2.3681, Validation Accuracy: 45.81%
2025-10-23 21:10:03,404 - INFO - Checkpoint saved
2025-10-23 21:10:04,168 - INFO - New best accuracy: 45.81% - Best model weights saved
2025-10-23 21:10:04,169 - INFO - Epoch 10 completed and logged.
2025-10-23 21:10:04,169 - INFO - ============================================================
2025-10-24 06:53:49,853 - INFO - EPOCH: 11/90 | LR: 0.096988
2025-10-24 07:31:43,355 - INFO - Epoch 11 Summary:
2025-10-24 07:31:43,356 - INFO -   Train Loss: 3.0965, Train Accuracy: 42.94%
2025-10-24 07:31:43,356 - INFO -   Validation Loss: 2.3992, Validation Accuracy: 45.36%
2025-10-24 07:31:44,173 - INFO - Checkpoint saved
2025-10-24 07:31:44,961 - INFO - New best accuracy: 45.36% - Best model weights saved
2025-10-24 07:31:44,961 - INFO - Epoch 11 completed and logged.
2025-10-24 07:31:44,961 - INFO - ============================================================
2025-10-24 07:31:44,961 - INFO - EPOCH: 12/90 | LR: 0.096363
2025-10-24 08:07:54,929 - INFO - Epoch 12 Summary:
2025-10-24 08:07:54,930 - INFO -   Train Loss: 1.8570, Train Accuracy: 43.25%
2025-10-24 08:07:54,930 - INFO -   Validation Loss: 2.3584, Validation Accuracy: 46.23%
2025-10-24 08:07:55,736 - INFO - Checkpoint saved
2025-10-24 08:07:56,526 - INFO - New best accuracy: 46.23% - Best model weights saved
2025-10-24 08:07:56,526 - INFO - Epoch 12 completed and logged.
2025-10-24 08:07:56,526 - INFO - ============================================================
2025-10-24 08:07:56,526 - INFO - EPOCH: 13/90 | LR: 0.095682
2025-10-24 08:44:08,650 - INFO - Epoch 13 Summary:
2025-10-24 08:44:08,651 - INFO -   Train Loss: 3.6882, Train Accuracy: 43.62%
2025-10-24 08:44:08,651 - INFO -   Validation Loss: 2.5619, Validation Accuracy: 42.99%
2025-10-24 08:44:09,480 - INFO - Checkpoint saved
2025-10-24 08:44:09,480 - INFO - Epoch 13 completed and logged.
2025-10-24 08:44:09,480 - INFO - ============================================================
2025-10-24 08:44:09,480 - INFO - EPOCH: 14/90 | LR: 0.094945
2025-10-24 09:20:22,012 - INFO - Epoch 14 Summary:
2025-10-24 09:20:22,013 - INFO -   Train Loss: 3.5924, Train Accuracy: 43.87%
2025-10-24 09:20:22,013 - INFO -   Validation Loss: 2.5039, Validation Accuracy: 43.64%
2025-10-24 09:20:22,820 - INFO - Checkpoint saved
2025-10-24 09:20:22,820 - INFO - Epoch 14 completed and logged.
2025-10-24 09:20:22,820 - INFO - ============================================================
2025-10-24 09:20:22,820 - INFO - EPOCH: 15/90 | LR: 0.094153
2025-10-24 09:56:36,987 - INFO - Epoch 15 Summary:
2025-10-24 09:56:36,988 - INFO -   Train Loss: 3.6536, Train Accuracy: 44.19%
2025-10-24 09:56:36,988 - INFO -   Validation Loss: 2.3391, Validation Accuracy: 46.18%
2025-10-24 09:56:37,827 - INFO - Checkpoint saved
2025-10-24 09:56:37,827 - INFO - Epoch 15 completed and logged.
2025-10-24 09:56:37,827 - INFO - ============================================================
2025-10-24 09:56:37,827 - INFO - EPOCH: 16/90 | LR: 0.093308
2025-10-24 10:32:52,202 - INFO - Epoch 16 Summary:
2025-10-24 10:32:52,202 - INFO -   Train Loss: 3.6952, Train Accuracy: 44.35%
2025-10-24 10:32:52,203 - INFO -   Validation Loss: 2.4695, Validation Accuracy: 44.12%
2025-10-24 10:32:53,060 - INFO - Checkpoint saved
2025-10-24 10:32:53,060 - INFO - Epoch 16 completed and logged.
2025-10-24 10:32:53,060 - INFO - ============================================================
2025-10-24 10:32:53,060 - INFO - EPOCH: 17/90 | LR: 0.092410
2025-10-24 11:09:09,802 - INFO - Epoch 17 Summary:
2025-10-24 11:09:09,803 - INFO -   Train Loss: 3.4605, Train Accuracy: 44.62%
2025-10-24 11:09:09,803 - INFO -   Validation Loss: 2.2380, Validation Accuracy: 48.51%
2025-10-24 11:09:10,615 - INFO - Checkpoint saved
2025-10-24 11:09:11,397 - INFO - New best accuracy: 48.51% - Best model weights saved
2025-10-24 11:09:11,397 - INFO - Epoch 17 completed and logged.
2025-10-24 11:09:11,397 - INFO - ============================================================
2025-10-24 11:09:11,397 - INFO - EPOCH: 18/90 | LR: 0.091460
2025-10-24 11:45:29,119 - INFO - Epoch 18 Summary:
2025-10-24 11:45:29,119 - INFO -   Train Loss: 3.4479, Train Accuracy: 44.86%
2025-10-24 11:45:29,120 - INFO -   Validation Loss: 2.2494, Validation Accuracy: 48.35%
2025-10-24 11:45:29,926 - INFO - Checkpoint saved
2025-10-24 11:45:29,926 - INFO - Epoch 18 completed and logged.
2025-10-24 11:45:29,926 - INFO - ============================================================
2025-10-24 11:45:29,926 - INFO - EPOCH: 19/90 | LR: 0.090460
2025-10-24 12:21:48,486 - INFO - Epoch 19 Summary:
2025-10-24 12:21:48,487 - INFO -   Train Loss: 2.7915, Train Accuracy: 45.03%
2025-10-24 12:21:48,487 - INFO -   Validation Loss: 2.2863, Validation Accuracy: 47.84%
2025-10-24 12:21:49,292 - INFO - Checkpoint saved
2025-10-24 12:21:49,292 - INFO - Epoch 19 completed and logged.
2025-10-24 12:21:49,292 - INFO - ============================================================
2025-10-24 12:21:49,292 - INFO - EPOCH: 20/90 | LR: 0.089411
2025-10-24 12:58:20,400 - INFO - Epoch 20 Summary:
2025-10-24 12:58:20,402 - INFO -   Train Loss: 2.9090, Train Accuracy: 45.25%
2025-10-24 12:58:20,402 - INFO -   Validation Loss: 2.2823, Validation Accuracy: 47.29%
2025-10-24 12:58:21,218 - INFO - Checkpoint saved
2025-10-24 12:58:21,218 - INFO - Epoch 20 completed and logged.
2025-10-24 12:58:21,218 - INFO - ============================================================
2025-10-24 15:14:55,016 - INFO - EPOCH: 21/90 | LR: 0.088314
2025-10-24 15:51:07,449 - INFO - Epoch 21 Summary:
2025-10-24 15:51:07,449 - INFO -   Train Loss: 3.0637, Train Accuracy: 45.47%
2025-10-24 15:51:07,449 - INFO -   Validation Loss: 2.2747, Validation Accuracy: 47.96%
2025-10-24 15:51:08,266 - INFO - Checkpoint saved
2025-10-24 15:51:09,042 - INFO - New best accuracy: 47.96% - Best model weights saved
2025-10-24 15:51:09,042 - INFO - Epoch 21 completed and logged.
2025-10-24 15:51:09,042 - INFO - ============================================================
2025-10-24 15:51:09,042 - INFO - EPOCH: 22/90 | LR: 0.087170
2025-10-24 16:27:18,475 - INFO - Epoch 22 Summary:
2025-10-24 16:27:18,475 - INFO -   Train Loss: 2.4963, Train Accuracy: 45.61%
2025-10-24 16:27:18,475 - INFO -   Validation Loss: 2.2060, Validation Accuracy: 49.40%
2025-10-24 16:27:19,292 - INFO - Checkpoint saved
2025-10-24 16:27:20,060 - INFO - New best accuracy: 49.40% - Best model weights saved
2025-10-24 16:27:20,060 - INFO - Epoch 22 completed and logged.
2025-10-24 16:27:20,060 - INFO - ============================================================
2025-10-24 16:27:20,060 - INFO - EPOCH: 23/90 | LR: 0.085981
2025-10-24 17:03:30,983 - INFO - Epoch 23 Summary:
2025-10-24 17:03:30,984 - INFO -   Train Loss: 2.8668, Train Accuracy: 45.85%
2025-10-24 17:03:30,984 - INFO -   Validation Loss: 2.3231, Validation Accuracy: 47.44%
2025-10-24 17:03:31,787 - INFO - Checkpoint saved
2025-10-24 17:03:31,788 - INFO - Epoch 23 completed and logged.
2025-10-24 17:03:31,788 - INFO - ============================================================
2025-10-24 17:03:31,788 - INFO - EPOCH: 24/90 | LR: 0.084748
2025-10-24 17:39:43,270 - INFO - Epoch 24 Summary:
2025-10-24 17:39:43,271 - INFO -   Train Loss: 3.1533, Train Accuracy: 46.07%
2025-10-24 17:39:43,271 - INFO -   Validation Loss: 2.3654, Validation Accuracy: 46.27%
2025-10-24 17:39:44,071 - INFO - Checkpoint saved
2025-10-24 17:39:44,071 - INFO - Epoch 24 completed and logged.
2025-10-24 17:39:44,071 - INFO - ============================================================
2025-10-24 17:39:44,071 - INFO - EPOCH: 25/90 | LR: 0.083473
2025-10-24 18:15:56,829 - INFO - Epoch 25 Summary:
2025-10-24 18:15:56,830 - INFO -   Train Loss: 5.0691, Train Accuracy: 46.25%
2025-10-24 18:15:56,830 - INFO -   Validation Loss: 2.1917, Validation Accuracy: 49.58%
2025-10-24 18:15:57,634 - INFO - Checkpoint saved
2025-10-24 18:15:58,413 - INFO - New best accuracy: 49.58% - Best model weights saved
2025-10-24 18:15:58,413 - INFO - Epoch 25 completed and logged.
2025-10-24 18:15:58,413 - INFO - ============================================================
2025-10-24 18:15:58,413 - INFO - EPOCH: 26/90 | LR: 0.082157
2025-10-24 18:52:12,680 - INFO - Epoch 26 Summary:
2025-10-24 18:52:12,681 - INFO -   Train Loss: 2.2636, Train Accuracy: 46.49%
2025-10-24 18:52:12,681 - INFO -   Validation Loss: 2.2354, Validation Accuracy: 48.72%
2025-10-24 18:52:13,501 - INFO - Checkpoint saved
2025-10-24 18:52:13,501 - INFO - Epoch 26 completed and logged.
2025-10-24 18:52:13,501 - INFO - ============================================================
2025-10-24 18:52:13,501 - INFO - EPOCH: 27/90 | LR: 0.080802
2025-10-24 19:28:28,446 - INFO - Epoch 27 Summary:
2025-10-24 19:28:28,447 - INFO -   Train Loss: 2.2679, Train Accuracy: 46.72%
2025-10-24 19:28:28,447 - INFO -   Validation Loss: 2.1414, Validation Accuracy: 50.52%
2025-10-24 19:28:29,248 - INFO - Checkpoint saved
2025-10-24 19:28:30,029 - INFO - New best accuracy: 50.52% - Best model weights saved
2025-10-24 19:28:30,030 - INFO - Epoch 27 completed and logged.
2025-10-24 19:28:30,030 - INFO - ============================================================
2025-10-24 19:28:30,030 - INFO - EPOCH: 28/90 | LR: 0.079410
2025-10-24 20:04:45,966 - INFO - Epoch 28 Summary:
2025-10-24 20:04:45,966 - INFO -   Train Loss: 2.3437, Train Accuracy: 46.88%
2025-10-24 20:04:45,966 - INFO -   Validation Loss: 2.1737, Validation Accuracy: 49.79%
2025-10-24 20:04:46,800 - INFO - Checkpoint saved
2025-10-24 20:04:46,800 - INFO - Epoch 28 completed and logged.
2025-10-24 20:04:46,801 - INFO - ============================================================
2025-10-24 20:04:46,801 - INFO - EPOCH: 29/90 | LR: 0.077982
2025-10-24 20:41:04,422 - INFO - Epoch 29 Summary:
2025-10-24 20:41:04,422 - INFO -   Train Loss: 2.4041, Train Accuracy: 47.12%
2025-10-24 20:41:04,423 - INFO -   Validation Loss: 2.1700, Validation Accuracy: 49.85%
2025-10-24 20:41:05,230 - INFO - Checkpoint saved
2025-10-24 20:41:05,230 - INFO - Epoch 29 completed and logged.
2025-10-24 20:41:05,230 - INFO - ============================================================
2025-10-24 20:41:05,230 - INFO - EPOCH: 30/90 | LR: 0.076519
2025-10-24 21:17:24,156 - INFO - Epoch 30 Summary:
2025-10-24 21:17:24,157 - INFO -   Train Loss: 3.4572, Train Accuracy: 47.40%
2025-10-24 21:17:24,157 - INFO -   Validation Loss: 2.1269, Validation Accuracy: 50.71%
2025-10-24 21:17:24,967 - INFO - Checkpoint saved
2025-10-24 21:17:25,748 - INFO - New best accuracy: 50.71% - Best model weights saved
2025-10-24 21:17:25,748 - INFO - Epoch 30 completed and logged.
2025-10-24 21:17:25,748 - INFO - ============================================================
2025-10-25 04:38:20,705 - INFO - EPOCH: 31/90 | LR: 0.075025
2025-10-25 05:14:32,495 - INFO - Epoch 31 Summary:
2025-10-25 05:14:32,497 - INFO -   Train Loss: 2.4893, Train Accuracy: 47.56%
2025-10-25 05:14:32,497 - INFO -   Validation Loss: 2.0743, Validation Accuracy: 51.76%
2025-10-25 05:14:33,308 - INFO - Checkpoint saved
2025-10-25 05:14:34,092 - INFO - New best accuracy: 51.76% - Best model weights saved
2025-10-25 05:14:34,092 - INFO - Epoch 31 completed and logged.
2025-10-25 05:14:34,092 - INFO - ============================================================
2025-10-25 05:14:34,092 - INFO - EPOCH: 32/90 | LR: 0.073500
2025-10-25 05:50:43,858 - INFO - Epoch 32 Summary:
2025-10-25 05:50:43,859 - INFO -   Train Loss: 2.7765, Train Accuracy: 47.81%
2025-10-25 05:50:43,859 - INFO -   Validation Loss: 2.1129, Validation Accuracy: 50.81%
2025-10-25 05:50:44,662 - INFO - Checkpoint saved
2025-10-25 05:50:44,663 - INFO - Epoch 32 completed and logged.
2025-10-25 05:50:44,663 - INFO - ============================================================
2025-10-25 05:50:44,663 - INFO - EPOCH: 33/90 | LR: 0.071947
2025-10-25 06:26:55,937 - INFO - Epoch 33 Summary:
2025-10-25 06:26:55,938 - INFO -   Train Loss: 3.0475, Train Accuracy: 48.06%
2025-10-25 06:26:55,938 - INFO -   Validation Loss: 2.2078, Validation Accuracy: 49.31%
2025-10-25 06:26:56,740 - INFO - Checkpoint saved
2025-10-25 06:26:56,740 - INFO - Epoch 33 completed and logged.
2025-10-25 06:26:56,740 - INFO - ============================================================
2025-10-25 06:26:56,740 - INFO - EPOCH: 34/90 | LR: 0.070366
2025-10-25 07:03:08,229 - INFO - Epoch 34 Summary:
2025-10-25 07:03:08,230 - INFO -   Train Loss: 3.3550, Train Accuracy: 48.31%
2025-10-25 07:03:08,230 - INFO -   Validation Loss: 2.1017, Validation Accuracy: 50.88%
2025-10-25 07:03:09,068 - INFO - Checkpoint saved
2025-10-25 07:03:09,068 - INFO - Epoch 34 completed and logged.
2025-10-25 07:03:09,068 - INFO - ============================================================
2025-10-25 07:03:09,068 - INFO - EPOCH: 35/90 | LR: 0.068762
2025-10-25 07:39:22,312 - INFO - Epoch 35 Summary:
2025-10-25 07:39:22,313 - INFO -   Train Loss: 2.1426, Train Accuracy: 48.58%
2025-10-25 07:39:22,313 - INFO -   Validation Loss: 2.0835, Validation Accuracy: 51.39%
2025-10-25 07:39:23,121 - INFO - Checkpoint saved
2025-10-25 07:39:23,121 - INFO - Epoch 35 completed and logged.
2025-10-25 07:39:23,121 - INFO - ============================================================
2025-10-25 07:39:23,121 - INFO - EPOCH: 36/90 | LR: 0.067134
2025-10-25 08:15:37,613 - INFO - Epoch 36 Summary:
2025-10-25 08:15:37,614 - INFO -   Train Loss: 2.6571, Train Accuracy: 48.86%
2025-10-25 08:15:37,614 - INFO -   Validation Loss: 2.0248, Validation Accuracy: 52.62%
2025-10-25 08:15:38,421 - INFO - Checkpoint saved
2025-10-25 08:15:39,204 - INFO - New best accuracy: 52.62% - Best model weights saved
2025-10-25 08:15:39,205 - INFO - Epoch 36 completed and logged.
2025-10-25 08:15:39,205 - INFO - ============================================================
2025-10-25 08:15:39,205 - INFO - EPOCH: 37/90 | LR: 0.065485
2025-10-25 08:51:55,266 - INFO - Epoch 37 Summary:
2025-10-25 08:51:55,267 - INFO -   Train Loss: 2.5989, Train Accuracy: 49.13%
2025-10-25 08:51:55,267 - INFO -   Validation Loss: 1.9885, Validation Accuracy: 53.32%
2025-10-25 08:51:56,068 - INFO - Checkpoint saved
2025-10-25 08:51:56,848 - INFO - New best accuracy: 53.32% - Best model weights saved
2025-10-25 08:51:56,848 - INFO - Epoch 37 completed and logged.
2025-10-25 08:51:56,848 - INFO - ============================================================
2025-10-25 08:51:56,848 - INFO - EPOCH: 38/90 | LR: 0.063818
2025-10-25 09:28:13,172 - INFO - Epoch 38 Summary:
2025-10-25 09:28:13,172 - INFO -   Train Loss: 2.4691, Train Accuracy: 49.47%
2025-10-25 09:28:13,172 - INFO -   Validation Loss: 2.0748, Validation Accuracy: 51.75%
2025-10-25 09:28:13,979 - INFO - Checkpoint saved
2025-10-25 09:28:13,979 - INFO - Epoch 38 completed and logged.
2025-10-25 09:28:13,979 - INFO - ============================================================
2025-10-25 09:28:13,979 - INFO - EPOCH: 39/90 | LR: 0.062134
2025-10-25 10:04:32,087 - INFO - Epoch 39 Summary:
2025-10-25 10:04:32,087 - INFO -   Train Loss: 3.8210, Train Accuracy: 49.69%
2025-10-25 10:04:32,088 - INFO -   Validation Loss: 2.0993, Validation Accuracy: 50.99%
2025-10-25 10:04:32,892 - INFO - Checkpoint saved
2025-10-25 10:04:32,892 - INFO - Epoch 39 completed and logged.
2025-10-25 10:04:32,892 - INFO - ============================================================
2025-10-25 10:04:32,892 - INFO - EPOCH: 40/90 | LR: 0.060435
2025-10-25 10:40:51,908 - INFO - Epoch 40 Summary:
2025-10-25 10:40:51,909 - INFO -   Train Loss: 2.2385, Train Accuracy: 49.96%
2025-10-25 10:40:51,909 - INFO -   Validation Loss: 1.9999, Validation Accuracy: 53.19%
2025-10-25 10:40:52,718 - INFO - Checkpoint saved
2025-10-25 10:40:52,988 - INFO - Epoch 40 completed and logged.
2025-10-25 10:40:52,988 - INFO - ============================================================
2025-10-25 19:38:47,050 - INFO - EPOCH: 41/90 | LR: 0.058724
2025-10-25 20:15:20,762 - INFO - Epoch 41 Summary:
2025-10-25 20:15:20,871 - INFO -   Train Loss: 3.3502, Train Accuracy: 50.25%
2025-10-25 20:15:20,871 - INFO -   Validation Loss: 1.9549, Validation Accuracy: 53.82%
2025-10-25 20:15:21,701 - INFO - Checkpoint saved
2025-10-25 20:15:21,839 - INFO - New best accuracy: 53.82% - Best model weights saved
2025-10-25 20:15:21,839 - INFO - Epoch 41 completed and logged.
2025-10-25 20:15:21,839 - INFO - ============================================================
2025-10-25 20:15:21,839 - INFO - EPOCH: 42/90 | LR: 0.057002
2025-10-25 20:51:32,652 - INFO - Epoch 42 Summary:
2025-10-25 20:51:32,653 - INFO -   Train Loss: 2.8370, Train Accuracy: 50.53%
2025-10-25 20:51:32,653 - INFO -   Validation Loss: 1.9579, Validation Accuracy: 54.07%
2025-10-25 20:51:33,467 - INFO - Checkpoint saved
2025-10-25 20:51:34,254 - INFO - New best accuracy: 54.07% - Best model weights saved
2025-10-25 20:51:34,255 - INFO - Epoch 42 completed and logged.
2025-10-25 20:51:34,255 - INFO - ============================================================
2025-10-25 20:51:34,255 - INFO - EPOCH: 43/90 | LR: 0.055271
2025-10-25 21:27:46,355 - INFO - Epoch 43 Summary:
2025-10-25 21:27:46,356 - INFO -   Train Loss: 2.5312, Train Accuracy: 50.91%
2025-10-25 21:27:46,356 - INFO -   Validation Loss: 1.9228, Validation Accuracy: 54.67%
2025-10-25 21:27:47,180 - INFO - Checkpoint saved
2025-10-25 21:27:47,960 - INFO - New best accuracy: 54.67% - Best model weights saved
2025-10-25 21:27:47,960 - INFO - Epoch 43 completed and logged.
2025-10-25 21:27:47,960 - INFO - ============================================================
2025-10-25 21:27:47,960 - INFO - EPOCH: 44/90 | LR: 0.053534
2025-10-25 22:04:00,520 - INFO - Epoch 44 Summary:
2025-10-25 22:04:00,521 - INFO -   Train Loss: 2.7122, Train Accuracy: 51.17%
2025-10-25 22:04:00,521 - INFO -   Validation Loss: 1.8622, Validation Accuracy: 55.75%
2025-10-25 22:04:01,343 - INFO - Checkpoint saved
2025-10-25 22:04:02,123 - INFO - New best accuracy: 55.75% - Best model weights saved
2025-10-25 22:04:02,123 - INFO - Epoch 44 completed and logged.
2025-10-25 22:04:02,123 - INFO - ============================================================
2025-10-25 22:04:02,123 - INFO - EPOCH: 45/90 | LR: 0.051793
2025-10-25 22:40:15,935 - INFO - Epoch 45 Summary:
2025-10-25 22:40:15,936 - INFO -   Train Loss: 3.2691, Train Accuracy: 51.45%
2025-10-25 22:40:15,936 - INFO -   Validation Loss: 1.8889, Validation Accuracy: 55.51%
2025-10-25 22:40:16,757 - INFO - Checkpoint saved
2025-10-25 22:40:16,773 - INFO - Epoch 45 completed and logged.
2025-10-25 22:40:16,773 - INFO - ============================================================
2025-10-25 22:40:16,773 - INFO - EPOCH: 46/90 | LR: 0.050050
2025-10-25 23:16:30,991 - INFO - Epoch 46 Summary:
2025-10-25 23:16:30,991 - INFO -   Train Loss: 2.6224, Train Accuracy: 51.81%
2025-10-25 23:16:30,991 - INFO -   Validation Loss: 1.8426, Validation Accuracy: 56.09%
2025-10-25 23:16:31,821 - INFO - Checkpoint saved
2025-10-25 23:16:32,603 - INFO - New best accuracy: 56.09% - Best model weights saved
2025-10-25 23:16:32,603 - INFO - Epoch 46 completed and logged.
2025-10-25 23:16:32,603 - INFO - ============================================================
2025-10-25 23:16:32,603 - INFO - EPOCH: 47/90 | LR: 0.048307
2025-10-25 23:52:47,847 - INFO - Epoch 47 Summary:
2025-10-25 23:52:47,939 - INFO -   Train Loss: 1.5646, Train Accuracy: 52.14%
2025-10-25 23:52:47,939 - INFO -   Validation Loss: 1.8534, Validation Accuracy: 56.15%
2025-10-25 23:52:48,758 - INFO - Checkpoint saved
2025-10-25 23:52:49,540 - INFO - New best accuracy: 56.15% - Best model weights saved
2025-10-25 23:52:49,540 - INFO - Epoch 47 completed and logged.
2025-10-25 23:52:49,540 - INFO - ============================================================
2025-10-25 23:52:49,540 - INFO - EPOCH: 48/90 | LR: 0.046566
2025-10-26 00:29:05,578 - INFO - Epoch 48 Summary:
2025-10-26 00:29:05,579 - INFO -   Train Loss: 3.5252, Train Accuracy: 52.52%
2025-10-26 00:29:05,579 - INFO -   Validation Loss: 1.8379, Validation Accuracy: 56.38%
2025-10-26 00:29:06,430 - INFO - Checkpoint saved
2025-10-26 00:29:07,205 - INFO - New best accuracy: 56.38% - Best model weights saved
2025-10-26 00:29:07,205 - INFO - Epoch 48 completed and logged.
2025-10-26 00:29:07,205 - INFO - ============================================================
2025-10-26 00:29:07,206 - INFO - EPOCH: 49/90 | LR: 0.044829
2025-10-26 01:05:23,486 - INFO - Epoch 49 Summary:
2025-10-26 01:05:23,487 - INFO -   Train Loss: 3.0376, Train Accuracy: 52.88%
2025-10-26 01:05:23,487 - INFO -   Validation Loss: 1.8105, Validation Accuracy: 57.25%
2025-10-26 01:05:24,325 - INFO - Checkpoint saved
2025-10-26 01:05:25,090 - INFO - New best accuracy: 57.25% - Best model weights saved
2025-10-26 01:05:25,090 - INFO - Epoch 49 completed and logged.
2025-10-26 01:05:25,090 - INFO - ============================================================
2025-10-26 01:05:25,090 - INFO - EPOCH: 50/90 | LR: 0.043098
2025-10-26 01:41:42,932 - INFO - Epoch 50 Summary:
2025-10-26 01:41:42,934 - INFO -   Train Loss: 1.9958, Train Accuracy: 53.22%
2025-10-26 01:41:42,934 - INFO -   Validation Loss: 1.7543, Validation Accuracy: 58.11%
2025-10-26 01:41:43,818 - INFO - Checkpoint saved
2025-10-26 01:41:44,583 - INFO - New best accuracy: 58.11% - Best model weights saved
2025-10-26 01:41:44,583 - INFO - Epoch 50 completed and logged.
2025-10-26 01:41:44,583 - INFO - ============================================================
2025-10-26 09:25:12,156 - INFO - EPOCH: 51/90 | LR: 0.041376
2025-10-26 10:01:24,329 - INFO - Epoch 51 Summary:
2025-10-26 10:01:24,330 - INFO -   Train Loss: 3.2798, Train Accuracy: 53.61%
2025-10-26 10:01:24,330 - INFO -   Validation Loss: 1.7680, Validation Accuracy: 57.72%
2025-10-26 10:01:25,153 - INFO - Checkpoint saved
2025-10-26 10:01:25,932 - INFO - New best accuracy: 57.72% - Best model weights saved
2025-10-26 10:01:25,932 - INFO - Epoch 51 completed and logged.
2025-10-26 10:01:25,932 - INFO - ============================================================
2025-10-26 10:01:25,932 - INFO - EPOCH: 52/90 | LR: 0.039665
2025-10-26 10:37:35,419 - INFO - Epoch 52 Summary:
2025-10-26 10:37:35,420 - INFO -   Train Loss: 2.2786, Train Accuracy: 54.07%
2025-10-26 10:37:35,420 - INFO -   Validation Loss: 1.7897, Validation Accuracy: 57.17%
2025-10-26 10:37:36,223 - INFO - Checkpoint saved
2025-10-26 10:37:36,223 - INFO - Epoch 52 completed and logged.
2025-10-26 10:37:36,223 - INFO - ============================================================
2025-10-26 10:37:36,223 - INFO - EPOCH: 53/90 | LR: 0.037966
2025-10-26 11:13:46,908 - INFO - Epoch 53 Summary:
2025-10-26 11:13:46,909 - INFO -   Train Loss: 3.3760, Train Accuracy: 54.40%
2025-10-26 11:13:46,909 - INFO -   Validation Loss: 1.7052, Validation Accuracy: 59.04%
2025-10-26 11:13:47,712 - INFO - Checkpoint saved
2025-10-26 11:13:48,494 - INFO - New best accuracy: 59.04% - Best model weights saved
2025-10-26 11:13:48,494 - INFO - Epoch 53 completed and logged.
2025-10-26 11:13:48,494 - INFO - ============================================================
2025-10-26 11:13:48,494 - INFO - EPOCH: 54/90 | LR: 0.036282
2025-10-26 11:50:00,442 - INFO - Epoch 54 Summary:
2025-10-26 11:50:00,445 - INFO -   Train Loss: 2.2370, Train Accuracy: 54.81%
2025-10-26 11:50:00,445 - INFO -   Validation Loss: 1.7549, Validation Accuracy: 58.14%
2025-10-26 11:50:01,262 - INFO - Checkpoint saved
2025-10-26 11:50:01,262 - INFO - Epoch 54 completed and logged.
2025-10-26 11:50:01,262 - INFO - ============================================================
2025-10-26 11:50:01,262 - INFO - EPOCH: 55/90 | LR: 0.034615
2025-10-26 12:26:13,510 - INFO - Epoch 55 Summary:
2025-10-26 12:26:13,511 - INFO -   Train Loss: 3.1238, Train Accuracy: 55.20%
2025-10-26 12:26:13,511 - INFO -   Validation Loss: 1.7581, Validation Accuracy: 57.94%
2025-10-26 12:26:14,313 - INFO - Checkpoint saved
2025-10-26 12:26:14,313 - INFO - Epoch 55 completed and logged.
2025-10-26 12:26:14,313 - INFO - ============================================================
2025-10-26 12:26:14,313 - INFO - EPOCH: 56/90 | LR: 0.032966
2025-10-26 13:02:28,373 - INFO - Epoch 56 Summary:
2025-10-26 13:02:28,373 - INFO -   Train Loss: 2.5399, Train Accuracy: 55.61%
2025-10-26 13:02:28,374 - INFO -   Validation Loss: 1.6570, Validation Accuracy: 60.06%
2025-10-26 13:02:29,185 - INFO - Checkpoint saved
2025-10-26 13:02:29,966 - INFO - New best accuracy: 60.06% - Best model weights saved
2025-10-26 13:02:29,966 - INFO - Epoch 56 completed and logged.
2025-10-26 13:02:29,966 - INFO - ============================================================
2025-10-26 13:02:29,966 - INFO - EPOCH: 57/90 | LR: 0.031338
2025-10-26 13:38:45,171 - INFO - Epoch 57 Summary:
2025-10-26 13:38:45,172 - INFO -   Train Loss: 2.8318, Train Accuracy: 56.10%
2025-10-26 13:38:45,172 - INFO -   Validation Loss: 1.7121, Validation Accuracy: 59.08%
2025-10-26 13:38:45,977 - INFO - Checkpoint saved
2025-10-26 13:38:45,977 - INFO - Epoch 57 completed and logged.
2025-10-26 13:38:45,977 - INFO - ============================================================
2025-10-26 13:38:45,977 - INFO - EPOCH: 58/90 | LR: 0.029734
2025-10-26 14:15:03,196 - INFO - Epoch 58 Summary:
2025-10-26 14:15:03,197 - INFO -   Train Loss: 1.6638, Train Accuracy: 56.55%
2025-10-26 14:15:03,197 - INFO -   Validation Loss: 1.6420, Validation Accuracy: 60.47%
2025-10-26 14:15:04,006 - INFO - Checkpoint saved
2025-10-26 14:15:04,780 - INFO - New best accuracy: 60.47% - Best model weights saved
2025-10-26 14:15:04,780 - INFO - Epoch 58 completed and logged.
2025-10-26 14:15:04,780 - INFO - ============================================================
2025-10-26 14:15:04,780 - INFO - EPOCH: 59/90 | LR: 0.028153
2025-10-26 14:51:22,590 - INFO - Epoch 59 Summary:
2025-10-26 14:51:22,591 - INFO -   Train Loss: 2.3088, Train Accuracy: 57.03%
2025-10-26 14:51:22,591 - INFO -   Validation Loss: 1.6065, Validation Accuracy: 61.11%
2025-10-26 14:51:23,393 - INFO - Checkpoint saved
2025-10-26 14:51:24,176 - INFO - New best accuracy: 61.11% - Best model weights saved
2025-10-26 14:51:24,176 - INFO - Epoch 59 completed and logged.
2025-10-26 14:51:24,176 - INFO - ============================================================
2025-10-26 14:51:24,176 - INFO - EPOCH: 60/90 | LR: 0.026600
2025-10-26 15:27:43,758 - INFO - Epoch 60 Summary:
2025-10-26 15:27:43,760 - INFO -   Train Loss: 2.7037, Train Accuracy: 57.50%
2025-10-26 15:27:43,760 - INFO -   Validation Loss: 1.5575, Validation Accuracy: 61.98%
2025-10-26 15:27:44,610 - INFO - Checkpoint saved
2025-10-26 15:27:45,370 - INFO - New best accuracy: 61.98% - Best model weights saved
2025-10-26 15:27:45,370 - INFO - Epoch 60 completed and logged.
2025-10-26 15:27:45,370 - INFO - ============================================================
2025-10-26 21:05:53,367 - INFO - EPOCH: 61/90 | LR: 0.025075
2025-10-26 21:42:05,352 - INFO - Epoch 61 Summary:
2025-10-26 21:42:05,352 - INFO -   Train Loss: 2.0725, Train Accuracy: 57.94%
2025-10-26 21:42:05,352 - INFO -   Validation Loss: 1.6061, Validation Accuracy: 61.12%
2025-10-26 21:42:05,978 - INFO - Checkpoint saved
2025-10-26 21:42:06,108 - INFO - New best accuracy: 61.12% - Best model weights saved
2025-10-26 21:42:06,108 - INFO - Epoch 61 completed and logged.
2025-10-26 21:42:06,108 - INFO - ============================================================
2025-10-26 21:42:06,108 - INFO - EPOCH: 62/90 | LR: 0.023581
2025-10-26 22:18:16,104 - INFO - Epoch 62 Summary:
2025-10-26 22:18:16,105 - INFO -   Train Loss: 2.1888, Train Accuracy: 58.44%
2025-10-26 22:18:16,105 - INFO -   Validation Loss: 1.5546, Validation Accuracy: 62.45%
2025-10-26 22:18:16,729 - INFO - Checkpoint saved
2025-10-26 22:18:17,016 - INFO - New best accuracy: 62.45% - Best model weights saved
2025-10-26 22:18:17,016 - INFO - Epoch 62 completed and logged.
2025-10-26 22:18:17,016 - INFO - ============================================================
2025-10-26 22:18:17,016 - INFO - EPOCH: 63/90 | LR: 0.022118
2025-10-26 22:54:28,287 - INFO - Epoch 63 Summary:
2025-10-26 22:54:28,288 - INFO -   Train Loss: 1.8107, Train Accuracy: 59.00%
2025-10-26 22:54:28,288 - INFO -   Validation Loss: 1.5553, Validation Accuracy: 62.20%
2025-10-26 22:54:28,878 - INFO - Checkpoint saved
2025-10-26 22:54:28,878 - INFO - Epoch 63 completed and logged.
2025-10-26 22:54:28,879 - INFO - ============================================================
2025-10-26 22:54:28,879 - INFO - EPOCH: 64/90 | LR: 0.020690
2025-10-26 23:30:40,909 - INFO - Epoch 64 Summary:
2025-10-26 23:30:40,910 - INFO -   Train Loss: 2.2998, Train Accuracy: 59.54%
2025-10-26 23:30:40,910 - INFO -   Validation Loss: 1.5308, Validation Accuracy: 62.72%
2025-10-26 23:30:41,521 - INFO - Checkpoint saved
2025-10-26 23:30:41,813 - INFO - New best accuracy: 62.72% - Best model weights saved
2025-10-26 23:30:41,813 - INFO - Epoch 64 completed and logged.
2025-10-26 23:30:41,813 - INFO - ============================================================
2025-10-26 23:30:41,813 - INFO - EPOCH: 65/90 | LR: 0.019298
2025-10-27 00:06:55,958 - INFO - Epoch 65 Summary:
2025-10-27 00:06:55,958 - INFO -   Train Loss: 2.3393, Train Accuracy: 60.02%
2025-10-27 00:06:55,958 - INFO -   Validation Loss: 1.4647, Validation Accuracy: 63.97%
2025-10-27 00:06:56,553 - INFO - Checkpoint saved
2025-10-27 00:06:56,795 - INFO - New best accuracy: 63.97% - Best model weights saved
2025-10-27 00:06:56,795 - INFO - Epoch 65 completed and logged.
2025-10-27 00:06:56,795 - INFO - ============================================================
2025-10-27 00:06:56,795 - INFO - EPOCH: 66/90 | LR: 0.017943
2025-10-27 00:43:12,899 - INFO - Epoch 66 Summary:
2025-10-27 00:43:12,900 - INFO -   Train Loss: 2.9043, Train Accuracy: 60.60%
2025-10-27 00:43:12,900 - INFO -   Validation Loss: 1.4814, Validation Accuracy: 63.87%
2025-10-27 00:43:13,502 - INFO - Checkpoint saved
2025-10-27 00:43:13,502 - INFO - Epoch 66 completed and logged.
2025-10-27 00:43:13,502 - INFO - ============================================================
2025-10-27 00:43:13,502 - INFO - EPOCH: 67/90 | LR: 0.016627
2025-10-27 01:19:29,997 - INFO - Epoch 67 Summary:
2025-10-27 01:19:29,998 - INFO -   Train Loss: 2.3802, Train Accuracy: 61.23%
2025-10-27 01:19:29,998 - INFO -   Validation Loss: 1.4028, Validation Accuracy: 65.30%
2025-10-27 01:19:30,565 - INFO - Checkpoint saved
2025-10-27 01:19:30,853 - INFO - New best accuracy: 65.30% - Best model weights saved
2025-10-27 01:19:30,854 - INFO - Epoch 67 completed and logged.
2025-10-27 01:19:30,854 - INFO - ============================================================
2025-10-27 01:19:30,854 - INFO - EPOCH: 68/90 | LR: 0.015352
2025-10-27 01:55:48,824 - INFO - Epoch 68 Summary:
2025-10-27 01:55:48,824 - INFO -   Train Loss: 2.2453, Train Accuracy: 61.85%
2025-10-27 01:55:48,824 - INFO -   Validation Loss: 1.3855, Validation Accuracy: 65.76%
2025-10-27 01:55:49,417 - INFO - Checkpoint saved
2025-10-27 01:55:49,665 - INFO - New best accuracy: 65.76% - Best model weights saved
2025-10-27 01:55:49,665 - INFO - Epoch 68 completed and logged.
2025-10-27 01:55:49,665 - INFO - ============================================================
2025-10-27 01:55:49,665 - INFO - EPOCH: 69/90 | LR: 0.014119
2025-10-27 02:32:08,495 - INFO - Epoch 69 Summary:
2025-10-27 02:32:08,496 - INFO -   Train Loss: 2.0793, Train Accuracy: 62.45%
2025-10-27 02:32:08,496 - INFO -   Validation Loss: 1.4133, Validation Accuracy: 65.18%
2025-10-27 02:32:09,059 - INFO - Checkpoint saved
2025-10-27 02:32:09,059 - INFO - Epoch 69 completed and logged.
2025-10-27 02:32:09,059 - INFO - ============================================================
2025-10-27 02:32:09,059 - INFO - EPOCH: 70/90 | LR: 0.012930
2025-10-27 03:08:29,147 - INFO - Epoch 70 Summary:
2025-10-27 03:08:29,148 - INFO -   Train Loss: 2.1797, Train Accuracy: 63.12%
2025-10-27 03:08:29,148 - INFO -   Validation Loss: 1.3720, Validation Accuracy: 66.28%
2025-10-27 03:08:29,733 - INFO - Checkpoint saved
2025-10-27 03:08:30,239 - INFO - New best accuracy: 66.28% - Best model weights saved
2025-10-27 03:08:30,240 - INFO - Epoch 70 completed and logged.
2025-10-27 03:08:30,240 - INFO - ============================================================
2025-10-27 06:40:59,507 - INFO - EPOCH: 71/90 | LR: 0.011786
2025-10-27 07:17:11,246 - INFO - Epoch 71 Summary:
2025-10-27 07:17:11,247 - INFO -   Train Loss: 1.2797, Train Accuracy: 63.71%
2025-10-27 07:17:11,247 - INFO -   Validation Loss: 1.3012, Validation Accuracy: 67.66%
2025-10-27 07:17:12,068 - INFO - Checkpoint saved
2025-10-27 07:17:12,847 - INFO - New best accuracy: 67.66% - Best model weights saved
2025-10-27 07:17:12,847 - INFO - Epoch 71 completed and logged.
2025-10-27 07:17:12,847 - INFO - ============================================================
2025-10-27 07:17:12,847 - INFO - EPOCH: 72/90 | LR: 0.010689
2025-10-27 07:53:22,763 - INFO - Epoch 72 Summary:
2025-10-27 07:53:22,764 - INFO -   Train Loss: 2.6017, Train Accuracy: 64.48%
2025-10-27 07:53:22,764 - INFO -   Validation Loss: 1.2766, Validation Accuracy: 68.20%
2025-10-27 07:53:23,581 - INFO - Checkpoint saved
2025-10-27 07:53:24,352 - INFO - New best accuracy: 68.20% - Best model weights saved
2025-10-27 07:53:24,352 - INFO - Epoch 72 completed and logged.
2025-10-27 07:53:24,352 - INFO - ============================================================
2025-10-27 07:53:24,352 - INFO - EPOCH: 73/90 | LR: 0.009640
2025-10-27 08:29:34,977 - INFO - Epoch 73 Summary:
2025-10-27 08:29:34,977 - INFO -   Train Loss: 2.7381, Train Accuracy: 65.12%
2025-10-27 08:29:34,977 - INFO -   Validation Loss: 1.2718, Validation Accuracy: 68.38%
2025-10-27 08:29:35,784 - INFO - Checkpoint saved
2025-10-27 08:29:36,562 - INFO - New best accuracy: 68.38% - Best model weights saved
2025-10-27 08:29:36,563 - INFO - Epoch 73 completed and logged.
2025-10-27 08:29:36,563 - INFO - ============================================================
2025-10-27 08:29:36,563 - INFO - EPOCH: 74/90 | LR: 0.008640
2025-10-27 09:05:48,479 - INFO - Epoch 74 Summary:
2025-10-27 09:05:48,480 - INFO -   Train Loss: 2.6757, Train Accuracy: 65.81%
2025-10-27 09:05:48,480 - INFO -   Validation Loss: 1.2656, Validation Accuracy: 68.65%
2025-10-27 09:05:49,331 - INFO - Checkpoint saved
2025-10-27 09:05:50,093 - INFO - New best accuracy: 68.65% - Best model weights saved
2025-10-27 09:05:50,094 - INFO - Epoch 74 completed and logged.
2025-10-27 09:05:50,094 - INFO - ============================================================
2025-10-27 09:05:50,094 - INFO - EPOCH: 75/90 | LR: 0.007690
2025-10-27 09:42:03,827 - INFO - Epoch 75 Summary:
2025-10-27 09:42:03,828 - INFO -   Train Loss: 1.9695, Train Accuracy: 66.62%
2025-10-27 09:42:03,828 - INFO -   Validation Loss: 1.2035, Validation Accuracy: 70.03%
2025-10-27 09:42:04,633 - INFO - Checkpoint saved
2025-10-27 09:42:05,414 - INFO - New best accuracy: 70.03% - Best model weights saved
2025-10-27 09:42:05,414 - INFO - Epoch 75 completed and logged.
2025-10-27 09:42:05,414 - INFO - ============================================================
2025-10-27 09:42:05,414 - INFO - EPOCH: 76/90 | LR: 0.006792
2025-10-27 10:18:20,581 - INFO - Epoch 76 Summary:
2025-10-27 10:18:20,582 - INFO -   Train Loss: 1.8192, Train Accuracy: 67.40%
2025-10-27 10:18:20,582 - INFO -   Validation Loss: 1.1885, Validation Accuracy: 70.40%
2025-10-27 10:18:21,400 - INFO - Checkpoint saved
2025-10-27 10:18:22,166 - INFO - New best accuracy: 70.40% - Best model weights saved
2025-10-27 10:18:22,166 - INFO - Epoch 76 completed and logged.
2025-10-27 10:18:22,166 - INFO - ============================================================
2025-10-27 10:18:22,166 - INFO - EPOCH: 77/90 | LR: 0.005947
2025-10-27 10:54:38,233 - INFO - Epoch 77 Summary:
2025-10-27 10:54:38,234 - INFO -   Train Loss: 1.3976, Train Accuracy: 68.17%
2025-10-27 10:54:38,234 - INFO -   Validation Loss: 1.1742, Validation Accuracy: 70.68%
2025-10-27 10:54:39,035 - INFO - Checkpoint saved
2025-10-27 10:54:39,819 - INFO - New best accuracy: 70.68% - Best model weights saved
2025-10-27 10:54:39,819 - INFO - Epoch 77 completed and logged.
2025-10-27 10:54:39,819 - INFO - ============================================================
2025-10-27 10:54:39,819 - INFO - EPOCH: 78/90 | LR: 0.005155
2025-10-27 11:30:56,676 - INFO - Epoch 78 Summary:
2025-10-27 11:30:56,677 - INFO -   Train Loss: 1.4738, Train Accuracy: 68.95%
2025-10-27 11:30:56,677 - INFO -   Validation Loss: 1.1606, Validation Accuracy: 71.14%
2025-10-27 11:30:57,479 - INFO - Checkpoint saved
2025-10-27 11:30:58,271 - INFO - New best accuracy: 71.14% - Best model weights saved
2025-10-27 11:30:58,271 - INFO - Epoch 78 completed and logged.
2025-10-27 11:30:58,271 - INFO - ============================================================
2025-10-27 11:30:58,272 - INFO - EPOCH: 79/90 | LR: 0.004418
2025-10-27 12:07:16,771 - INFO - Epoch 79 Summary:
2025-10-27 12:07:16,772 - INFO -   Train Loss: 1.6743, Train Accuracy: 69.86%
2025-10-27 12:07:16,772 - INFO -   Validation Loss: 1.1084, Validation Accuracy: 72.12%
2025-10-27 12:07:17,578 - INFO - Checkpoint saved
2025-10-27 12:07:18,358 - INFO - New best accuracy: 72.12% - Best model weights saved
2025-10-27 12:07:18,358 - INFO - Epoch 79 completed and logged.
2025-10-27 12:07:18,358 - INFO - ============================================================
2025-10-27 12:07:18,358 - INFO - EPOCH: 80/90 | LR: 0.003737
2025-10-27 12:43:37,536 - INFO - Epoch 80 Summary:
2025-10-27 12:43:37,537 - INFO -   Train Loss: 2.0871, Train Accuracy: 70.68%
2025-10-27 12:43:37,537 - INFO -   Validation Loss: 1.0846, Validation Accuracy: 72.60%
2025-10-27 12:43:38,356 - INFO - Checkpoint saved
2025-10-27 12:43:39,134 - INFO - New best accuracy: 72.60% - Best model weights saved
2025-10-27 12:43:39,134 - INFO - Epoch 80 completed and logged.
2025-10-27 12:43:39,134 - INFO - ============================================================
2025-10-27 16:42:06,347 - INFO - EPOCH: 81/90 | LR: 0.003112
2025-10-27 17:18:18,983 - INFO - Epoch 81 Summary:
2025-10-27 17:18:18,984 - INFO -   Train Loss: 2.3448, Train Accuracy: 71.58%
2025-10-27 17:18:18,984 - INFO -   Validation Loss: 1.0605, Validation Accuracy: 73.20%
2025-10-27 17:18:19,794 - INFO - Checkpoint saved
2025-10-27 17:18:19,924 - INFO - New best accuracy: 73.20% - Best model weights saved
2025-10-27 17:18:19,924 - INFO - Epoch 81 completed and logged.
2025-10-27 17:18:19,924 - INFO - ============================================================
2025-10-27 17:18:19,924 - INFO - EPOCH: 82/90 | LR: 0.002545
2025-10-27 17:54:29,325 - INFO - Epoch 82 Summary:
2025-10-27 17:54:29,326 - INFO -   Train Loss: 1.7945, Train Accuracy: 72.47%
2025-10-27 17:54:29,326 - INFO -   Validation Loss: 1.0441, Validation Accuracy: 73.48%
2025-10-27 17:54:30,126 - INFO - Checkpoint saved
2025-10-27 17:54:30,918 - INFO - New best accuracy: 73.48% - Best model weights saved
2025-10-27 17:54:30,918 - INFO - Epoch 82 completed and logged.
2025-10-27 17:54:30,918 - INFO - ============================================================
2025-10-27 17:54:30,918 - INFO - EPOCH: 83/90 | LR: 0.002035
2025-10-27 18:30:42,281 - INFO - Epoch 83 Summary:
2025-10-27 18:30:42,282 - INFO -   Train Loss: 1.9466, Train Accuracy: 73.29%
2025-10-27 18:30:42,282 - INFO -   Validation Loss: 1.0245, Validation Accuracy: 74.09%
2025-10-27 18:30:43,088 - INFO - Checkpoint saved
2025-10-27 18:30:43,871 - INFO - New best accuracy: 74.09% - Best model weights saved
2025-10-27 18:30:43,871 - INFO - Epoch 83 completed and logged.
2025-10-27 18:30:43,871 - INFO - ============================================================
2025-10-27 18:30:43,871 - INFO - EPOCH: 84/90 | LR: 0.001584
2025-10-27 19:06:56,052 - INFO - Epoch 84 Summary:
2025-10-27 19:06:56,052 - INFO -   Train Loss: 1.5110, Train Accuracy: 74.12%
2025-10-27 19:06:56,053 - INFO -   Validation Loss: 1.0122, Validation Accuracy: 74.56%
2025-10-27 19:06:56,858 - INFO - Checkpoint saved
2025-10-27 19:06:57,645 - INFO - New best accuracy: 74.56% - Best model weights saved
2025-10-27 19:06:57,646 - INFO - Epoch 84 completed and logged.
2025-10-27 19:06:57,646 - INFO - ============================================================
2025-10-27 19:06:57,646 - INFO - EPOCH: 85/90 | LR: 0.001192
2025-10-27 19:43:10,797 - INFO - Epoch 85 Summary:
2025-10-27 19:43:10,798 - INFO -   Train Loss: 1.7164, Train Accuracy: 74.86%
2025-10-27 19:43:10,798 - INFO -   Validation Loss: 0.9918, Validation Accuracy: 74.96%
2025-10-27 19:43:11,620 - INFO - Checkpoint saved
2025-10-27 19:43:12,388 - INFO - New best accuracy: 74.96% - Best model weights saved
2025-10-27 19:43:12,388 - INFO - Epoch 85 completed and logged.
2025-10-27 19:43:12,388 - INFO - ============================================================
2025-10-27 19:43:12,388 - INFO - EPOCH: 86/90 | LR: 0.000859
2025-10-27 20:19:26,920 - INFO - Epoch 86 Summary:
2025-10-27 20:19:26,921 - INFO -   Train Loss: 1.0278, Train Accuracy: 75.57%
2025-10-27 20:19:26,921 - INFO -   Validation Loss: 0.9750, Validation Accuracy: 75.29%
2025-10-27 20:19:27,727 - INFO - Checkpoint saved
2025-10-27 20:19:28,519 - INFO - New best accuracy: 75.29% - Best model weights saved
2025-10-27 20:19:28,519 - INFO - Epoch 86 completed and logged.
2025-10-27 20:19:28,519 - INFO - ============================================================
2025-10-27 20:19:28,519 - INFO - EPOCH: 87/90 | LR: 0.000586
2025-10-27 20:55:44,259 - INFO - Epoch 87 Summary:
2025-10-27 20:55:44,260 - INFO -   Train Loss: 1.9779, Train Accuracy: 76.19%
2025-10-27 20:55:44,260 - INFO -   Validation Loss: 0.9654, Validation Accuracy: 75.60%
2025-10-27 20:55:45,062 - INFO - Checkpoint saved
2025-10-27 20:55:45,843 - INFO - New best accuracy: 75.60% - Best model weights saved
2025-10-27 20:55:45,844 - INFO - Epoch 87 completed and logged.
2025-10-27 20:55:45,844 - INFO - ============================================================
2025-10-27 20:55:45,844 - INFO - EPOCH: 88/90 | LR: 0.000374
2025-10-27 21:32:03,020 - INFO - Epoch 88 Summary:
2025-10-27 21:32:03,021 - INFO -   Train Loss: 2.1364, Train Accuracy: 76.66%
2025-10-27 21:32:03,021 - INFO -   Validation Loss: 0.9560, Validation Accuracy: 75.80%
2025-10-27 21:32:03,826 - INFO - Checkpoint saved
2025-10-27 21:32:04,607 - INFO - New best accuracy: 75.80% - Best model weights saved
2025-10-27 21:32:04,608 - INFO - Epoch 88 completed and logged.
2025-10-27 21:32:04,608 - INFO - ============================================================
2025-10-27 21:32:04,608 - INFO - EPOCH: 89/90 | LR: 0.000222
2025-10-27 22:08:21,888 - INFO - Epoch 89 Summary:
2025-10-27 22:08:21,889 - INFO -   Train Loss: 1.6449, Train Accuracy: 76.98%
2025-10-27 22:08:21,889 - INFO -   Validation Loss: 0.9522, Validation Accuracy: 76.01%
2025-10-27 22:08:22,698 - INFO - Checkpoint saved
2025-10-27 22:08:23,481 - INFO - New best accuracy: 76.01% - Best model weights saved
2025-10-27 22:08:23,482 - INFO - Epoch 89 completed and logged.
2025-10-27 22:08:23,482 - INFO - ============================================================
2025-10-27 22:08:23,482 - INFO - EPOCH: 90/90 | LR: 0.000130
2025-10-27 22:44:42,454 - INFO - Epoch 90 Summary:
2025-10-27 22:44:42,455 - INFO -   Train Loss: 1.5460, Train Accuracy: 77.23%
2025-10-27 22:44:42,455 - INFO -   Validation Loss: 0.9459, Validation Accuracy: 76.15%
2025-10-27 22:44:43,289 - INFO - Checkpoint saved
2025-10-27 22:44:44,061 - INFO - New best accuracy: 76.15% - Best model weights saved
2025-10-27 22:44:44,061 - INFO - Epoch 90 completed and logged.
2025-10-27 22:44:44,061 - INFO - ============================================================
2025-10-27 22:44:44,061 - INFO - Training completed!
2025-10-27 22:44:44,061 - INFO - Best validation accuracy achieved: 76.15%

