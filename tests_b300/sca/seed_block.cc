  if (p.batch == 0) return {dq, dk, dv, dprev_k, dprev_v};
  // Comparison base: the old COMPLETE generic schedule, not a phase-driver plan.
  // The current parser returns a request, not an enum. Read it once and keep the
  // old allocation/fold condition; det-off unsplit dQ still uses unordered atomics.
  const BackwardRequest request = BackwardArmOptIn();
  TORCH_CHECK(!request.require_native,
              "seam-inertness generic base cannot satisfy XATTN_SCA_BWD_ARM=native");
  const bool split = request.arm == BackwardArm::kSplit;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());
  at::Tensor key_tile_range;
  if (p.row_lo != nullptr) {
    p.key_tile_count = (p.total_length + 63) / 64;
    key_tile_range = at::empty({p.batch, p.key_tile_count, 2}, q.options().dtype(at::kInt));
    p.key_tile_range = key_tile_range.data_ptr<int32_t>();
  }
  at::Tensor dq_accum;
  if (!deterministic && !split) {
    dq_accum = at::zeros(dq.sizes(), q.options().dtype(at::kFloat));
    p.dq_accum = dq_accum.data_ptr<float>();
  }
  LaunchTileBounds(p, stream);
  if (p.row_lo != nullptr) LaunchKeyTileRange(p, stream);
  LaunchBackward(p, stream);
  if (!deterministic && !split) dq.copy_(dq_accum.mul_(p.scale));
  g_backward_launches.fetch_add(1, std::memory_order_relaxed);
  return {dq, dk, dv, dprev_k, dprev_v};
