// No stored copy of the seam lives at this path.
//
// The pair's seam side is `csrc/flash_sca/native/host.cc` itself, so a copy of it
// here could only drift from the file it stands for -- and a pair that swaps a
// stale copy measures the wrong tree twice. The base side is produced by
// `seam_inertness_probe.py swap --to base`, which needs exactly one stored body:
// `seed_block.cc`, the generic body the seam falls back to.
