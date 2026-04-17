# RF-DETR Nano: TRT Layer → CUDA Kernel Mapping

Each row maps a TRT layer (from `IProfiler`) to its actual CUDA kernel(s)
(from nsys `CUPTI_ACTIVITY_KIND_KERNEL` trace, iteration 12).

**Key:**
- nsys #1-5 are PyTorch preprocessing kernels (outside TRT)
- nsys #194 is `fused_postprocess_kernel` (outside TRT)
- Some TRT layers emit 2 CUDA kernels (main + split-k reduction)
- Zero-cost TRT layers (sync/memory ops) have no CUDA kernel

## Summary by Block

| Block | TRT Time (ms) | nsys Time (us) | Layers |
|-------|---------------|----------------|--------|
| Input/Output | 0.047 (0.8%) | 39 (1.2%) | 3 |
| Backbone (ViT Encoder) | 4.461 (72.2%) | 2456 (73.6%) | 95 |
| Projector (CSP Neck) | 0.452 (7.3%) | 227 (6.8%) | 22 |
| Encoder Output (Two-Stage) | 0.229 (3.7%) | 123 (3.7%) | 10 |
| Decoder | 0.887 (14.4%) | 444 (13.3%) | 38 |
| Detection Heads | 0.072 (1.2%) | 36 (1.1%) | 3 |
| TRT Internal | 0.033 (0.5%) | 10 (0.3%) | 22 |
| **Total** | **6.181** | **3335** | **193** |

## Complete Mapping (Execution Order)

| TRT# | TRT Time | nsys# | nsys Time | Model Path | Description | CUDA Kernel |
|------|----------|-------|-----------|------------|-------------|-------------|
| 1 | 0.0294ms | 6 | 17.8us | `forward(samples)` | FP32→FP16 input cast | `__myl_Cast_0x88a9fc905715f4bf12322bd14c6ba089` |
| 2 | 0.1409ms | 7 | 79.8us | `backbone[0].encoder.encoder.embeddings.patch_embeddings.projection` | Patch embedding Conv2d(3, 384, 16, 16) | `sm75_xmma_fprop_implicit_gemm_indexed_wo_smem_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x32x64_stage1_warpsize4x1x1_g1_tensor16x8x8_alignc8_execute_kernel_trt` |
| 3 | 0.0151ms | 8 | 8.4us | `backbone[0].encoder.encoder.embeddings` | ViT: position embedding add + reshape | `__myl_TranConcAddSlicReplSlicReshTranReshMoveConcCastMeanSubMulMean_0x4d2ae1947518f17642a7315ed364811e` |
| 4 | 0.0156ms | 9 | 8.8us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_AddSqrtDivMulCastMulAddReshReshTran_0x3bd0e47e6d9763a38e7103a8dfce83c4` |
| 5 | 0.0686ms | 10 | 40.4us | `backbone[0].encoder.encoder.encoder.layer[0].attention.attention.{query,key,value}` | ViT L0: fused QKV projection (3 × Linear(384, 384)) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 6 | 0.0342ms | 11 | 18.6us | `backbone[0].encoder.encoder.encoder.layer[*].attention` | ViT: multi-head attention softmax(QK^T)V | `_gemm_mha_v2_0x865c16f51c59c6f8d5af8f4d8f92b6d0` |
| 7 | 0.0315ms | 12 | 17.8us | `backbone[0].encoder.encoder.encoder.layer[0].attention.output.dense` | ViT L0: attn output proj + gate (fused) | `sm75_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x64x64_stage1_warpsize2x2x1_tensor16x8x8_fused` |
| 8 | 0.0123ms | 13 | 7.1us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_ReshTranReshAddReshCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0xa9941c24f12dda940577fbd4d6543828` |
| 9 | 0.0799ms | 14 | 46.0us | `backbone[0].encoder.encoder.encoder.layer[*].mlp.fc1` | ViT: MLP fc1 Linear(384, 1536) | `sm75_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_fused` |
| 10 | 0.0762ms | 15,16 | 45.2us | `backbone[0].encoder.encoder.encoder.layer[0].mlp.fc2` | ViT L0: MLP fc2 Linear(1536, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt + split-k` |
| 11 | 0.0144ms | 17 | 9.0us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_MulAddCastMeanSubMulMean_0xbbf607cadbb06df395bae5a4cab888ca` |
| 12 | 0.0143ms | 18 | 7.8us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_AddSqrtDivMulCastMulAddReshTran_0x4ec093f9472b9380d2c9628d6bad847f` |
| 13 | 0.0696ms | 19 | 38.9us | `backbone[0].encoder.encoder.encoder.layer[1].attention.attention.{query,key,value}` | ViT L1: fused QKV projection (3 × Linear(384, 384)) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 14 | 0.0328ms | 20 | 18.7us | `backbone[0].encoder.encoder.encoder.layer[*].attention` | ViT: multi-head attention softmax(QK^T)V | `_gemm_mha_v2_0x865c16f51c59c6f8d5af8f4d8f92b6d0` |
| 15 | 0.0324ms | 21 | 17.6us | `backbone[0].encoder.encoder.encoder.layer[1].attention.output.dense` | ViT L1: attention output projection Linear(384, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize64x64x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 16 | 0.0126ms | 22 | 6.3us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_MulReshTranReshAddCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0x728fd2f56c52addb349ae7df61e1afba` |
| 17 | 0.0803ms | 23 | 46.3us | `backbone[0].encoder.encoder.encoder.layer[*].mlp.fc1` | ViT: MLP fc1 Linear(384, 1536) | `sm75_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_fused` |
| 18 | 0.0771ms | 24,25 | 44.9us | `backbone[0].encoder.encoder.encoder.layer[1].mlp.fc2` | ViT L1: MLP fc2 Linear(1536, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt + split-k` |
| 19 | 0.0138ms | 26 | 7.9us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_MulAddCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0x90cd16860312fa3c9753ce2464cfb6f5` |
| 20 | 0.0683ms | 27 | 39.1us | `backbone[0].encoder.encoder.encoder.layer[2].attention.attention.{query,key,value}` | ViT L2: fused QKV projection (3 × Linear(384, 384)) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 21 | 0.0330ms | 28 | 18.7us | `backbone[0].encoder.encoder.encoder.layer[*].attention` | ViT: multi-head attention softmax(QK^T)V | `_gemm_mha_v2_0x865c16f51c59c6f8d5af8f4d8f92b6d0` |
| 22 | 0.0348ms | 29 | 18.1us | `backbone[0].encoder.encoder.encoder.layer[2].attention.output.dense` | ViT L2: attention output projection Linear(384, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize64x64x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 23 | 0.0143ms | 30 | 6.4us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_MulAddCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0x90cd16860312fa3c9753ce2464cfb6f5` |
| 24 | 0.0820ms | 31 | 46.0us | `backbone[0].encoder.encoder.encoder.layer[*].mlp.fc1` | ViT: MLP fc1 Linear(384, 1536) | `sm75_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_fused` |
| 25 | 0.0819ms | 32,33 | 44.4us | `backbone[0].encoder.encoder.encoder.layer[2].mlp.fc2` | ViT L2: MLP fc2 Linear(1536, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt + split-k` |
| 26 | 0.0184ms | 34 | 9.2us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_MulAddReshTranCastMeanSubMulMean_0xa7b7a32cb180b0e73b3064d64e877ca2` |
| 27 | 0.0133ms | 35 | 5.2us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_ReshCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0xf5550b1853ea077bfd3c9df31f1deb12` |
| 28 | 0.0715ms | 36 | 39.3us | `backbone[0].encoder.encoder.encoder.layer[3].attention.attention.{query,key,value}` | ViT L3: fused QKV projection (3 × Linear(384, 384)) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 29 | 0.0944ms | 37 | 52.2us | `backbone[0].encoder.encoder.encoder.layer[*].attention` | ViT: multi-head attention softmax(QK^T)V | `_gemm_mha_v2_0xa5968bddd7596481813c8d0b8a101bd3` |
| 30 | 0.0348ms | 38 | 17.9us | `backbone[0].encoder.encoder.encoder.layer[3].attention.output.dense` | ViT L3: attention output projection Linear(384, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize64x64x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 31 | 0.0143ms | 39 | 6.5us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_ReshMulAddReshCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0x7bca40fb3bd1d3bc62cb840987bcd417` |
| 32 | 0.0812ms | 40 | 46.3us | `backbone[0].encoder.encoder.encoder.layer[*].mlp.fc1` | ViT: MLP fc1 Linear(384, 1536) | `sm75_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_fused` |
| 33 | 0.0809ms | 41,42 | 44.8us | `backbone[0].encoder.encoder.encoder.layer[3].mlp.fc2` | ViT L3: MLP fc2 Linear(1536, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt + split-k` |
| 34 | 0.0164ms | 43 | 8.9us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_MulAddCastMeanSubMulMean_0xbbf607cadbb06df395bae5a4cab888ca` |
| 35 | 0.0175ms | 44 | 7.8us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_AddSqrtDivMulCastMulAddReshTran_0x4ec093f9472b9380d2c9628d6bad847f` |
| 36 | 0.0716ms | 45 | 39.4us | `backbone[0].encoder.encoder.encoder.layer[4].attention.attention.{query,key,value}` | ViT L4: fused QKV projection (3 × Linear(384, 384)) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 37 | 0.0352ms | 46 | 18.5us | `backbone[0].encoder.encoder.encoder.layer[*].attention` | ViT: multi-head attention softmax(QK^T)V | `_gemm_mha_v2_0x865c16f51c59c6f8d5af8f4d8f92b6d0` |
| 38 | 0.0345ms | 47 | 17.6us | `backbone[0].encoder.encoder.encoder.layer[4].attention.output.dense` | ViT L4: attention output projection Linear(384, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize64x64x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 39 | 0.0128ms | 48 | 6.4us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_MulReshTranReshAddCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0x728fd2f56c52addb349ae7df61e1afba` |
| 40 | 0.0830ms | 49 | 46.6us | `backbone[0].encoder.encoder.encoder.layer[*].mlp.fc1` | ViT: MLP fc1 Linear(384, 1536) | `sm75_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_fused` |
| 41 | 0.0816ms | 50,51 | 45.1us | `backbone[0].encoder.encoder.encoder.layer[4].mlp.fc2` | ViT L4: MLP fc2 Linear(1536, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt + split-k` |
| 42 | 0.0141ms | 52 | 8.4us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_MulAddCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0x90cd16860312fa3c9753ce2464cfb6f5` |
| 43 | 0.0686ms | 53 | 39.2us | `backbone[0].encoder.encoder.encoder.layer[5].attention.attention.{query,key,value}` | ViT L5: fused QKV projection (3 × Linear(384, 384)) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 44 | 0.0367ms | 54 | 18.8us | `backbone[0].encoder.encoder.encoder.layer[*].attention` | ViT: multi-head attention softmax(QK^T)V | `_gemm_mha_v2_0x865c16f51c59c6f8d5af8f4d8f92b6d0` |
| 45 | 0.0331ms | 55 | 18.0us | `backbone[0].encoder.encoder.encoder.layer[5].attention.output.dense` | ViT L5: attention output projection Linear(384, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize64x64x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 46 | 0.0143ms | 56 | 6.7us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_MulAddCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0x90cd16860312fa3c9753ce2464cfb6f5` |
| 47 | 0.0818ms | 57 | 45.9us | `backbone[0].encoder.encoder.encoder.layer[*].mlp.fc1` | ViT: MLP fc1 Linear(384, 1536) | `sm75_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_fused` |
| 48 | 0.0823ms | 58,59 | 45.2us | `backbone[0].encoder.encoder.encoder.layer[5].mlp.fc2` | ViT L5: MLP fc2 Linear(1536, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt + split-k` |
| 49 | 0.0176ms | 60 | 9.2us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_MulAddReshTranCastMeanSubMulMean_0xa7b7a32cb180b0e73b3064d64e877ca2` |
| 50 | 0.0139ms | 61 | 5.5us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_ReshCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0xf5550b1853ea077bfd3c9df31f1deb12` |
| 51 | 0.0695ms | 62 | 39.9us | `backbone[0].encoder.encoder.encoder.layer[6].attention.attention.{query,key,value}` | ViT L6: fused QKV projection (3 × Linear(384, 384)) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 52 | 0.0954ms | 63 | 52.4us | `backbone[0].encoder.encoder.encoder.layer[*].attention` | ViT: multi-head attention softmax(QK^T)V | `_gemm_mha_v2_0xa5968bddd7596481813c8d0b8a101bd3` |
| 53 | 0.0364ms | 64 | 17.1us | `backbone[0].encoder.encoder.encoder.layer[6].attention.output.dense` | ViT L6: attention output projection Linear(384, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize64x64x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 54 | 0.0147ms | 65 | 5.8us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_ReshMulAddReshCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0x7bca40fb3bd1d3bc62cb840987bcd417` |
| 55 | 0.0825ms | 66 | 45.7us | `backbone[0].encoder.encoder.encoder.layer[*].mlp.fc1` | ViT: MLP fc1 Linear(384, 1536) | `sm75_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_fused` |
| 56 | 0.0801ms | 67,68 | 45.0us | `backbone[0].encoder.encoder.encoder.layer[6].mlp.fc2` | ViT L6: MLP fc2 Linear(1536, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt + split-k` |
| 57 | 0.0167ms | 69 | 8.7us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_MulAddCastMeanSubMulMean_0xbbf607cadbb06df395bae5a4cab888ca` |
| 58 | 0.0164ms | 70 | 7.9us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_AddSqrtDivMulCastMulAddReshTran_0x4ec093f9472b9380d2c9628d6bad847f` |
| 59 | 0.0696ms | 71 | 39.5us | `backbone[0].encoder.encoder.encoder.layer[7].attention.attention.{query,key,value}` | ViT L7: fused QKV projection (3 × Linear(384, 384)) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 60 | 0.0348ms | 72 | 18.4us | `backbone[0].encoder.encoder.encoder.layer[*].attention` | ViT: multi-head attention softmax(QK^T)V | `_gemm_mha_v2_0x865c16f51c59c6f8d5af8f4d8f92b6d0` |
| 61 | 0.0361ms | 73 | 18.0us | `backbone[0].encoder.encoder.encoder.layer[7].attention.output.dense` | ViT L7: attention output projection Linear(384, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize64x64x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 62 | 0.0143ms | 74 | 6.4us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_MulReshTranReshAddCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0x728fd2f56c52addb349ae7df61e1afba` |
| 63 | 0.0830ms | 75 | 45.9us | `backbone[0].encoder.encoder.encoder.layer[*].mlp.fc1` | ViT: MLP fc1 Linear(384, 1536) | `sm75_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_fused` |
| 64 | 0.0820ms | 76,77 | 44.6us | `backbone[0].encoder.encoder.encoder.layer[7].mlp.fc2` | ViT L7: MLP fc2 Linear(1536, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt + split-k` |
| 65 | 0.0143ms | 78 | 8.5us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_MulAddCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0x90cd16860312fa3c9753ce2464cfb6f5` |
| 66 | 0.0696ms | 79 | 38.8us | `backbone[0].encoder.encoder.encoder.layer[8].attention.attention.{query,key,value}` | ViT L8: fused QKV projection (3 × Linear(384, 384)) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 67 | 0.0365ms | 80 | 18.8us | `backbone[0].encoder.encoder.encoder.layer[*].attention` | ViT: multi-head attention softmax(QK^T)V | `_gemm_mha_v2_0x865c16f51c59c6f8d5af8f4d8f92b6d0` |
| 68 | 0.0335ms | 81 | 17.9us | `backbone[0].encoder.encoder.encoder.layer[8].attention.output.dense` | ViT L8: attention output projection Linear(384, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize64x64x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 69 | 0.0138ms | 82 | 6.5us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_MulAddCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0x90cd16860312fa3c9753ce2464cfb6f5` |
| 70 | 0.0823ms | 83 | 46.2us | `backbone[0].encoder.encoder.encoder.layer[*].mlp.fc1` | ViT: MLP fc1 Linear(384, 1536) | `sm75_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_fused` |
| 71 | 0.0806ms | 84,85 | 45.3us | `backbone[0].encoder.encoder.encoder.layer[8].mlp.fc2` | ViT L8: MLP fc2 Linear(1536, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt + split-k` |
| 72 | 0.0173ms | 86 | 9.0us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_MulAddReshTranCastMeanSubMulMean_0xa7b7a32cb180b0e73b3064d64e877ca2` |
| 73 | 0.0117ms | 87 | 5.5us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_ReshCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0xf5550b1853ea077bfd3c9df31f1deb12` |
| 74 | 0.0716ms | 88 | 40.6us | `backbone[0].encoder.encoder.encoder.layer[9].attention.attention.{query,key,value}` | ViT L9: fused QKV projection (3 × Linear(384, 384)) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 75 | 0.0942ms | 89 | 52.1us | `backbone[0].encoder.encoder.encoder.layer[*].attention` | ViT: multi-head attention softmax(QK^T)V | `_gemm_mha_v2_0xa5968bddd7596481813c8d0b8a101bd3` |
| 76 | 0.0340ms | 90 | 17.0us | `backbone[0].encoder.encoder.encoder.layer[9].attention.output.dense` | ViT L9: attention output projection Linear(384, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize64x64x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 77 | 0.0144ms | 91 | 6.3us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_ReshMulAddReshCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0x7bca40fb3bd1d3bc62cb840987bcd417` |
| 78 | 0.0821ms | 92 | 46.2us | `backbone[0].encoder.encoder.encoder.layer[*].mlp.fc1` | ViT: MLP fc1 Linear(384, 1536) | `sm75_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_fused` |
| 79 | 0.0809ms | 93,94 | 45.0us | `backbone[0].encoder.encoder.encoder.layer[9].mlp.fc2` | ViT L9: MLP fc2 Linear(1536, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt + split-k` |
| 80 | 0.0164ms | 95 | 9.3us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_MulAddCastMeanSubMulMean_0xbbf607cadbb06df395bae5a4cab888ca` |
| 81 | 0.0164ms | 96 | 8.1us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_AddSqrtDivMulCastMulAddReshTran_0x4ec093f9472b9380d2c9628d6bad847f` |
| 82 | 0.0717ms | 97 | 39.4us | `backbone[0].encoder.encoder.encoder.layer[10].attention.attention.{query,key,value}` | ViT L10: fused QKV projection (3 × Linear(384, 384)) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 83 | 0.0360ms | 98 | 18.8us | `backbone[0].encoder.encoder.encoder.layer[*].attention` | ViT: multi-head attention softmax(QK^T)V | `_gemm_mha_v2_0x865c16f51c59c6f8d5af8f4d8f92b6d0` |
| 84 | 0.0348ms | 99 | 17.8us | `backbone[0].encoder.encoder.encoder.layer[10].attention.output.dense` | ViT L10: attention output projection Linear(384, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize64x64x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 85 | 0.0138ms | 100 | 6.1us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_MulReshTranReshAddCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0x728fd2f56c52addb349ae7df61e1afba` |
| 86 | 0.0828ms | 101 | 46.8us | `backbone[0].encoder.encoder.encoder.layer[*].mlp.fc1` | ViT: MLP fc1 Linear(384, 1536) | `sm75_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_fused` |
| 87 | 0.0802ms | 102,103 | 44.8us | `backbone[0].encoder.encoder.encoder.layer[10].mlp.fc2` | ViT L10: MLP fc2 Linear(1536, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt + split-k` |
| 88 | 0.0142ms | 104 | 7.9us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_MulAddCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0x90cd16860312fa3c9753ce2464cfb6f5` |
| 89 | 0.0699ms | 105 | 40.2us | `backbone[0].encoder.encoder.encoder.layer[11].attention.attention.{query,key,value}` | ViT L11: fused QKV projection (3 × Linear(384, 384)) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 90 | 0.0362ms | 106 | 18.7us | `backbone[0].encoder.encoder.encoder.layer[*].attention` | ViT: multi-head attention softmax(QK^T)V | `_gemm_mha_v2_0x865c16f51c59c6f8d5af8f4d8f92b6d0` |
| 91 | 0.0342ms | 107 | 18.0us | `backbone[0].encoder.encoder.encoder.layer[11].attention.output.dense` | ViT L11: attention output projection Linear(384, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize64x64x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 92 | 0.0131ms | 108 | 6.6us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_MulAddCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0x90cd16860312fa3c9753ce2464cfb6f5` |
| 93 | 0.0840ms | 109 | 46.3us | `backbone[0].encoder.encoder.encoder.layer[*].mlp.fc1` | ViT: MLP fc1 Linear(384, 1536) | `sm75_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_fused` |
| 94 | 0.0797ms | 110,111 | 45.2us | `backbone[0].encoder.encoder.encoder.layer[11].mlp.fc2` | ViT L11: MLP fc2 Linear(1536, 384) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt + split-k` |
| 95 | 0.0146ms | 112 | 7.8us | `backbone[0].encoder.encoder.encoder.layer[*]` | ViT: fused LayerNorm / residual | `__myl_MulAddCastMeanSubMulMean_0x39b6e2e536e3906b478eb49e9f42ecaa` |
| 96 | 0.0353ms | 113 | 24.4us | `backbone[0].encoder.encoder.layernorm` | ViT: final LayerNorm + slice features at [3,6,9,12] | `__myl_AddSqrtDivMulCastMulAddReshTranReshSlicReshTranReshMoveTranAddSqrtDivMulCastMulAddReshTranEtc_0xcd5074b0ae6c012e111b7248c7ad1c43` |
| 97 | 0.0594ms | 114,115 | 34.9us | `backbone[0].projector.stages[0][0].cv1.conv` | C2f input Conv2d(1536, 512, 1) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x64x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt + split-k` |
| 98 | 0.0103ms | 116 | 4.3us | `backbone[0].projector.stages[0][0].*.bn` | CSP projector: BatchNorm (mean+var) | `__myl_TranCastMeanSubMulMean_0xe4bc4a6d1660d58cadedc724a0e08b78` |
| 99 | 0.0119ms | 117 | 4.8us | `backbone[0].projector.stages[0][0].cv1.bn + act` | CSP projector: BN + SiLU + slice | `__myl_AddSqrtDivMulCastMulAddTranSiluSlic_0xbbc725457976ac7789b8378651a0a42a` |
| 100 | 0.0372ms | 118 | 20.5us | `backbone[0].projector.stages[0][0].m[0].cv1.conv` | C2f bottleneck 0 conv1 Conv2d(256, 256, 3) | `sm75_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize64x32x64_stage1_warpsize2x1x2_g1_tensor16x8x8_t1r3s3_execute_kernel_trt` |
| 101 | 0.0101ms | 119 | 3.1us | `backbone[0].projector.stages[0][0].*.bn` | CSP projector: BatchNorm (mean+var) | `__myl_TranCastMeanSubMulMean_0xfdc590c97896954d7e1d4e642ec57bc7` |
| 102 | 0.0116ms | 120 | 4.3us | `backbone[0].projector.stages[0][0].*.bn + act` | CSP projector: BN + SiLU | `__myl_AddSqrtDivMulCastMulAddTranSilu_0xebe2f6d6d50cbde251549082aa3d6d20` |
| 103 | 0.0338ms | 121 | 19.6us | `backbone[0].projector.stages[0][0].m[0].cv2.conv` | C2f bottleneck 0 conv2 Conv2d(256, 256, 3) | `sm75_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize64x32x64_stage1_warpsize2x1x2_g1_tensor16x8x8_t1r3s3_execute_kernel_trt` |
| 104 | 0.0103ms | 122 | 3.2us | `backbone[0].projector.stages[0][0].*.bn` | CSP projector: BatchNorm (mean+var) | `__myl_TranCastMeanSubMulMean_0xfdc590c97896954d7e1d4e642ec57bc7` |
| 105 | 0.0113ms | 123 | 4.3us | `backbone[0].projector.stages[0][0].*.bn + act` | CSP projector: BN + SiLU | `__myl_AddSqrtDivMulCastMulAddTranSilu_0xebe2f6d6d50cbde251549082aa3d6d20` |
| 106 | 0.0363ms | 124 | 19.5us | `backbone[0].projector.stages[0][0].m[1].cv1.conv` | C2f bottleneck 1 conv1 Conv2d(256, 256, 3) | `sm75_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize64x32x64_stage1_warpsize2x1x2_g1_tensor16x8x8_t1r3s3_execute_kernel_trt` |
| 107 | 0.0091ms | 125 | 3.2us | `backbone[0].projector.stages[0][0].*.bn` | CSP projector: BatchNorm (mean+var) | `__myl_TranCastMeanSubMulMean_0xfdc590c97896954d7e1d4e642ec57bc7` |
| 108 | 0.0120ms | 126 | 4.3us | `backbone[0].projector.stages[0][0].*.bn + act` | CSP projector: BN + SiLU | `__myl_AddSqrtDivMulCastMulAddTranSilu_0xebe2f6d6d50cbde251549082aa3d6d20` |
| 109 | 0.0353ms | 127 | 19.6us | `backbone[0].projector.stages[0][0].m[1].cv2.conv` | C2f bottleneck 1 conv2 Conv2d(256, 256, 3) | `sm75_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize64x32x64_stage1_warpsize2x1x2_g1_tensor16x8x8_t1r3s3_execute_kernel_trt` |
| 110 | 0.0084ms | 128 | 3.0us | `backbone[0].projector.stages[0][0].*.bn` | CSP projector: BatchNorm (mean+var) | `__myl_TranCastMeanSubMulMean_0xfdc590c97896954d7e1d4e642ec57bc7` |
| 111 | 0.0102ms | 129 | 4.3us | `backbone[0].projector.stages[0][0].*.bn + act` | CSP projector: BN + SiLU | `__myl_AddSqrtDivMulCastMulAddTranSilu_0xebe2f6d6d50cbde251549082aa3d6d20` |
| 112 | 0.0351ms | 130 | 19.6us | `backbone[0].projector.stages[0][0].m[2].cv1.conv` | C2f bottleneck 2 conv1 Conv2d(256, 256, 3) | `sm75_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize64x32x64_stage1_warpsize2x1x2_g1_tensor16x8x8_t1r3s3_execute_kernel_trt` |
| 113 | 0.0097ms | 131 | 3.2us | `backbone[0].projector.stages[0][0].*.bn` | CSP projector: BatchNorm (mean+var) | `__myl_TranCastMeanSubMulMean_0xfdc590c97896954d7e1d4e642ec57bc7` |
| 114 | 0.0106ms | 132 | 4.4us | `backbone[0].projector.stages[0][0].*.bn + act` | CSP projector: BN + SiLU | `__myl_AddSqrtDivMulCastMulAddTranSilu_0xebe2f6d6d50cbde251549082aa3d6d20` |
| 115 | 0.0353ms | 133 | 19.5us | `backbone[0].projector.stages[0][0].m[2].cv2.conv` | C2f bottleneck 2 conv2 Conv2d(256, 256, 3) | `sm75_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize64x32x64_stage1_warpsize2x1x2_g1_tensor16x8x8_t1r3s3_execute_kernel_trt` |
| 116 | 0.0084ms | 134 | 3.2us | `backbone[0].projector.stages[0][0].*.bn` | CSP projector: BatchNorm (mean+var) | `__myl_TranCastMeanSubMulMean_0xfdc590c97896954d7e1d4e642ec57bc7` |
| 117 | 0.0136ms | 135 | 6.0us | `backbone[0].projector.stages[0][0].cv2.bn + act` | CSP projector: BN + SiLU + concat | `__myl_SlicAddSqrtDivMulCastMulAddTranSiluConc_0x3de1245dbc02efc4ad18d1cfc6a35eab` |
| 118 | 0.0317ms | 136 | 18.0us | `backbone[0].projector.stages[0][0].cv2.conv` | C2f output Conv2d(768, 256, 1) | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize64x64x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 119 | 0 | — | 0 | `(TRT internal)` | TRT: graph entry/exit signal | `(no CUDA kernel)` |
| 120 | 0 | — | 0 | `(TRT internal)` | TRT: graph entry/exit signal | `(no CUDA kernel)` |
| 121 | 0.0492ms | 137 | 4.1us | `transformer.decoder.layers[0].self_attn` | Decoder L0: self-attn score Q@K^T | `__myl_TranCastMeanSubMulMean_0xe4bc4a6d1660d58cadedc724a0e08b78` |
| 122 | 0 | — | 0 | `(TRT internal)` | TRT: internal data movement | `(no CUDA kernel)` |
| 123 | 0.0120ms | 138 | 26.3us | `transformer.decoder.layers[*].norm*` | Decoder: LayerNorm (mean+var) | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize64x32x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 124 | 0.0211ms | 139 | 18.0us | `transformer.decoder.layers[*].norm*` | Decoder: LayerNorm (affine) + transpose | `__myl_AddSqrtDivMulCastMulAddTran_0x268627a053fbd7edd97239887b6a9e2a` |
| 125 | 0.0100ms | 140 | 2.7us | `transformer.decoder` | Decoder: SiLU activation | `__myl_Silu_0x565e9cc4ce03c1fc628e6489689fdec7` |
| 126 | 0.0106ms | 141 | 3.4us | `transformer.decoder` | Decoder: transpose | `__myl_Tran_0xb77f233ca34b346c768b4603d93f4486` |
| 127 | 0.0108ms | 142 | 4.2us | `transformer.enc_output_norm[0]` | Encoder output LayerNorm (mean+var) | `__myl_CastMeanSubMulMean_0xba823e1c0f8920d8ab86e7f7ea5f4488` |
| 128 | 0.0205ms | 143 | 9.6us | `transformer.decoder.layers[*].norm*` | Decoder: LayerNorm (affine) + transpose | `__myl_AddSqrtDivMulCastMulAddTran_0x4a76fc2d5b6049b0c4a20cbb61d740e4` |
| 129 | 0 | — | 0 | `(TRT internal)` | TRT: internal data movement | `(no CUDA kernel)` |
| 130 | 0 | — | 0 | `(TRT internal)` | TRT: internal data movement | `(no CUDA kernel)` |
| 131 | 0.0659ms | 144 | 4.1us | `transformer.decoder.layers[0,1].cross_attn.value_proj` | Decoder L0+L1: fused cross-attn value proj (2 × Linear(256, 256)) | `__myl_ReshTran_0x163c1e6a4b4046d0396dc1641b59c477` |
| 132 | 0 | — | 0 | `(TRT internal)` | TRT: internal data movement | `(no CUDA kernel)` |
| 133 | 0.0123ms | 145 | 38.8us | `transformer.decoder` | Decoder: transpose | `sm75_xmma_gemm_f16f16_f16f16_f16_nt_n_tilesize64x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 134 | 0.0200ms | 146 | 8.9us | `transformer` | Two-stage: spatial position encoding generation | `__myl_IotaCastCastAddReshReplReshReshReplReshConcReshAddMulReplConcReshLtGtrAndNotCastSlicSlicAddEtc_0x62a0537e92c8ac64409ab778bb443e65` |
| 135 | 0.0273ms | 147 | 14.4us | `transformer.enc_output[0]` | Encoder output projection Linear(256, 256) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize64x64x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 136 | 0.0107ms | 148 | 4.6us | `transformer.enc_output_norm[0]` | Encoder output LayerNorm (affine) | `__myl_CastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0x017141b19ce784784d2addc9d0aa8215` |
| 137 | 0 | — | 0 | `(TRT internal)` | TRT: internal data movement | `(no CUDA kernel)` |
| 138 | 0 | — | 0 | `(TRT internal)` | TRT: internal data movement | `(no CUDA kernel)` |
| 139 | 0.0369ms | 149 | 23.6us | `transformer.enc_out_class_embed[0]` | Encoder output class head Linear(256, 81) | `sm75_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize32x32x64_stage1_warpsize2x2x1_tensor16x8x8_aligna2_alignc2_execute_kernel_trt` |
| 140 | 0.0096ms | 150 | 20.3us | `transformer (two-stage)` | Two-stage: max score + reshape | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize64x64x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 141 | 0.0337ms | 151 | 2.8us | `transformer (two-stage)` | Two-stage: top-k proposal selection | `__myl_MaxrResh_0xe975b7e8cbefe9ac58cd5a186b9cfee3` |
| 142 | 0 | — | 0 | `(TRT internal)` | TRT: internal data movement | `(no CUDA kernel)` |
| 143 | 0.0339ms | 152 | 14.6us | `transformer.enc_out_bbox_embed[0].layers[0]` | Encoder output bbox MLP layer 0 Linear(256, 256) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize64x64x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 144 | 0.0236ms | 153 | 18.0us | `transformer.enc_out_bbox_embed[0].layers[1]` | Encoder output bbox MLP layer 1 Linear(256, 256) | `__myl_Topk_0x612a857e959121f4ba1df388c9ccb495` |
| 145 | 0.0184ms | 154 | 8.2us | `(TRT internal)` | TRT: internal data movement | `sm75_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize32x32x64_stage1_warpsize2x2x1_tensor16x8x8_aligna4_alignc4_execute_kernel_trt` |
| 146 | 0 | — | 0 | `(TRT internal)` | TRT: internal data movement | `(no CUDA kernel)` |
| 147 | 0.0225ms | 155 | 11.7us | `transformer` | Two-stage: reference point init from proposals | `__myl_SeleSlicSlicCastReshCastReplExpMulMulAddConcGathSlicSlicMulMulAddConcSlicSlicReshSlicSlicEtc_0x3c816a3d33d006a7b0064160156f2043` |
| 148 | 0.0260ms | 156 | 12.9us | `transformer.decoder.ref_point_head.layers[0]` | Reference point head Linear(512, 256) | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize64x32x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 149 | 0.0187ms | 157 | 8.8us | `transformer.decoder.ref_point_head.layers[1]` | Reference point head Linear(256, 256) | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize64x32x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 150 | 0.0091ms | 158 | 3.5us | `transformer.decoder` | Decoder: position encoding add to queries | `__myl_Add_0x7041be9f6f90732fdd3cda6776fcba92` |
| 151 | 0.0241ms | 159 | 11.6us | `transformer.decoder.layers[0].self_attn.in_proj` | Decoder L0: self-attn fused Q+K projection | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize64x64x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 152 | 0.0083ms | 160 | 3.2us | `transformer.decoder.layers[*].self_attn` | Decoder: self-attn Q/K reshape for multi-head | `__myl_TranReshTranReshMove_0x86b5c62c00dbc22802588a2df978ec85` |
| 153 | 0 | — | 0 | `(TRT internal)` | TRT: internal data movement | `(no CUDA kernel)` |
| 154 | 0.0451ms | 161 | 24.3us | `transformer.decoder.layers[*].self_attn` | Decoder: self-attn softmax(QK^T)V | `_gemm_mha_v2_0x715576b49ad8f039496a9c6e01f09182` |
| 155 | 0.0185ms | 162 | 8.9us | `transformer.decoder.layers[0].self_attn.out_proj` | Decoder L0: self-attn output proj Linear(256, 256) | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize64x32x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 156 | 0.0123ms | 163 | 4.4us | `transformer.decoder.layers[*].norm1` | Decoder: self-attn residual + LayerNorm | `__myl_AddCastMeanSubMulMeanAddSqrtDivMulCastMulAddReshAdd_0x88399a9d0107bda74f0c8807f8283b69` |
| 157 | 0.0163ms | 164 | 8.4us | `transformer.decoder.layers[0].cross_attn.{attention_weights, sampling_offsets}` | Decoder L0: deformable attn fused weights+offsets | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize32x32x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 158 | 0.0081ms | 165 | 2.5us | `transformer.decoder.layers[*].cross_attn` | Decoder: deformable attn offset computation | `__myl_ReshMulMulMulAddMulAddReshTranResh_0x18af4aa187bad06c6a1fac7153ea101e` |
| 159 | 0 | — | 0 | `(TRT internal)` | TRT: internal data movement | `(no CUDA kernel)` |
| 160 | 0.0334ms | 166 | 17.5us | `transformer.decoder.layers[*].cross_attn` | Decoder: deformable attn bilinear sampling | `__myl_AddReshReshSlicSlicMaxSubExpSlicSlicAddDivMulTranReshGridCastMulSlicSlicAdd_0x053711bf8b1f3587f62925d825449368` |
| 161 | 0.0245ms | 167 | 12.6us | `transformer.decoder.layers[0].cross_attn.output_proj` | Decoder L0: cross-attn output proj Linear(256, 256) | `sm75_xmma_gemm_f16f16_f16f32_f32_nt_n_tilesize64x64x64_stage1_warpsize2x2x1_tensor16x8x8_aligna4_alignc4_execute_kernel_trt` |
| 162 | 0.0082ms | 168 | 3.7us | `transformer.decoder.layers[*].norm2` | Decoder: cross-attn residual + LayerNorm | `__myl_AddCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0xa4deb5483aee644da51ccffb19ed0431` |
| 163 | 0.0369ms | 169 | 20.6us | `transformer.decoder.layers[0].linear1` | Decoder L0: FFN fc1 Linear(256, 2048) | `trt_turing_h1688gemm_256x64_ldg8_relu_stages_32x1_nn_v1` |
| 164 | 0.0523ms | 170,171 | 28.4us | `transformer.decoder.layers[0].linear2` | Decoder L0: FFN fc2 Linear(2048, 256) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize64x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt + split-k` |
| 165 | 0.0102ms | 172 | 4.1us | `transformer.decoder.layers[*].norm3` | Decoder: FFN residual + LayerNorm | `__myl_AddCastMeanSubMulMeanAddSqrtDivMulCastMulAddAdd_0x56bb505f984a5ea2148a941968c0aa72` |
| 166 | 0 | — | 0 | `(TRT internal)` | TRT: internal data movement | `(no CUDA kernel)` |
| 167 | 0 | — | 0 | `(TRT internal)` | TRT: internal data movement | `(no CUDA kernel)` |
| 168 | 0.0287ms | 173 | 11.8us | `transformer.decoder.layers[1].self_attn` | Decoder L1: self-attn score Q@K^T | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize64x32x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 169 | 0 | — | 0 | `(TRT internal)` | TRT: internal data movement | `(no CUDA kernel)` |
| 170 | 0.0287ms | 174 | 11.9us | `transformer.decoder.layers[1].self_attn.in_proj` | Decoder L1: self-attn fused Q+K projection | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize64x64x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 171 | 0.0082ms | 175 | 3.2us | `transformer.decoder.layers[*].self_attn` | Decoder: self-attn Q/K reshape for multi-head | `__myl_TranReshTranReshMove_0x86b5c62c00dbc22802588a2df978ec85` |
| 172 | 0 | — | 0 | `(TRT internal)` | TRT: internal data movement | `(no CUDA kernel)` |
| 173 | 0.0448ms | 176 | 24.0us | `transformer.decoder.layers[*].self_attn` | Decoder: self-attn softmax(QK^T)V | `_gemm_mha_v2_0x715576b49ad8f039496a9c6e01f09182` |
| 174 | 0.0186ms | 177 | 8.9us | `transformer.decoder.layers[1].self_attn.out_proj` | Decoder L1: self-attn output proj Linear(256, 256) | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize64x32x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 175 | 0.0108ms | 178 | 4.4us | `transformer.decoder.layers[*].norm1` | Decoder: self-attn residual + LayerNorm | `__myl_AddCastMeanSubMulMeanAddSqrtDivMulCastMulAddReshAdd_0xa5643579801d28787799e26038f027bc` |
| 176 | 0.0156ms | 179 | 7.3us | `transformer.decoder.layers[1].cross_attn.{attention_weights, sampling_offsets}` | Decoder L1: deformable attn fused weights+offsets | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize32x32x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 177 | 0.0081ms | 180 | 2.7us | `transformer.decoder.layers[*].cross_attn` | Decoder: deformable attn offset computation | `__myl_ReshMulMulMulAddMulAddReshTranResh_0x18af4aa187bad06c6a1fac7153ea101e` |
| 178 | 0.0340ms | 181 | 17.4us | `transformer.decoder.layers[*].cross_attn` | Decoder: deformable attn bilinear sampling | `__myl_AddReshReshSlicSlicMaxSubExpSlicSlicAddDivMulTranReshGridCastMulSlicSlicAdd_0x053711bf8b1f3587f62925d825449368` |
| 179 | 0.0234ms | 182 | 12.9us | `transformer.decoder.layers[1].cross_attn.output_proj` | Decoder L1: cross-attn output proj Linear(256, 256) | `sm75_xmma_gemm_f16f16_f16f32_f32_nt_n_tilesize64x64x64_stage1_warpsize2x2x1_tensor16x8x8_aligna4_alignc4_execute_kernel_trt` |
| 180 | 0.0092ms | 183 | 3.7us | `transformer.decoder.layers[*].norm2` | Decoder: cross-attn residual + LayerNorm | `__myl_AddCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0xa4deb5483aee644da51ccffb19ed0431` |
| 181 | 0.0380ms | 184 | 19.9us | `transformer.decoder.layers[1].linear1` | Decoder L1: FFN fc1 Linear(256, 2048) | `trt_turing_h1688gemm_256x64_ldg8_relu_stages_32x1_nn_v1` |
| 182 | 0.0533ms | 185,186 | 28.9us | `transformer.decoder.layers[1].linear2` | Decoder L1: FFN fc2 Linear(2048, 256) | `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize64x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt + split-k` |
| 183 | 0.0123ms | 187 | 4.9us | `transformer.decoder.norm` | Decoder: final double LayerNorm | `__myl_AddCastMeanSubMulMeanAddSqrtDivMulCastMulAddCastMeanSubMulMeanAddSqrtDivMulCastMulAdd_0x70b15df9cc38dbaa4bbba33dd3a69efd` |
| 184 | 0 | — | 0 | `(TRT internal)` | TRT: internal data movement | `(no CUDA kernel)` |
| 185 | 0.0228ms | 188 | 15.5us | `bbox_embed.layers[0]` | Bbox MLP layer 0 Linear(256, 256) | `sm75_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize32x32x64_stage1_warpsize2x2x1_tensor16x8x8_aligna2_alignc2_execute_kernel_trt` |
| 186 | 0.0205ms | 189 | 12.5us | `bbox_embed.layers[1]` | Bbox MLP layer 1 Linear(256, 256) | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize64x32x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 187 | 0.0145ms | 190 | 2.3us | `(TRT internal)` | TRT: internal data movement | `__myl_Cast_0x21164c816e851035ec7b973ae922138c` |
| 188 | 0.0073ms | 191 | 18.7us | `(output)` | Output: FP16→FP32 cast | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize64x32x64_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt` |
| 189 | 0 | — | 0 | `(TRT internal)` | TRT: internal data movement | `(no CUDA kernel)` |
| 190 | 0.0285ms | 192 | 7.7us | `class_embed` | Classification head Linear(256, 81) | `sm75_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize32x32x64_stage1_warpsize2x2x1_tensor16x8x8_aligna4_alignc4_execute_kernel_trt` |
| 191 | 0.0105ms | 193 | 2.7us | `(output)` | Output: FP16→FP32 cast | `__myl_ExpMulMulAddConcCast_0x819d89c6d8b0c1adabfa5196f28a9394` |
| 192 | 0 | — | 0 | `(TRT internal)` | TRT: graph entry/exit signal | `(no CUDA kernel)` |
| 193 | 0 | — | 0 | `(TRT internal)` | TRT: graph entry/exit signal | `(no CUDA kernel)` |

## Non-TRT Kernels

| nsys# | Time (us) | Description | CUDA Kernel |
|-------|-----------|-------------|-------------|
| 1 | 10.8 | PyTorch preprocessing | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda` |
| 2 | 18.7 | PyTorch preprocessing | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu` |
| 3 | 5.7 | PyTorch preprocessing | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BUnaryFunctor` |
| 4 | 10.1 | PyTorch preprocessing | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kerne` |
| 5 | 10.8 | PyTorch preprocessing | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kerne` |
| 194 | 9.9 | Inference postprocessing | `fused_postprocess_kernel` |