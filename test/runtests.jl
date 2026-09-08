include("synth.jl")
include("tssos.jl")

# The test below is not suitable for CI: uses Mosek requiring a license and attempts to solve large ill-conditioned random SDP.
# Run instead with
#   julia --project=test test/large.jl
#include("large.jl")
