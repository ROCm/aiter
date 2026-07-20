# FlyDSL HSTU Backward Kernel Dev - General Notes

This document records all general information on where to find common references and the constraints you must follow.

#### Info

* We are running here on a MI300, this is the target architecture. 
* Our main repo is Aiter (that you can find at `/workspaces/git/meta/aiter`)
* Our colleague repo, Aiter fork, can be found at `/workspaces/git/meta/robin_aiter/`. He has implemented the FlyDSL HSTU forward kernel.
* Our theory document is at `2026-07-07_HSTU_theory.md`
* Our working python environment is at `/workspaces/git/meta/aiter/flydsl_venv`
* Our plan is at `/workspaces/git/meta/aiter/docs/2026-07-07_HSTU_backward_plan.md`. This is a living document, to be update as we implement features listed there.
* Our performance log is at `/workspaces/git/meta/aiter/docs/HSTU_backward_optimization_log.md`. Please update that document whenever new benchmark deemed to be put there are run. (e.g. avoid adding "intermediate" bench when debugging, but add when validating a hypotheses about an optimization change). Specify the flydsl run time, TFLOPS, and if compared with Triton also add triton run time, TFLOPS and speedup (if applicable) 

#### Constraints
 
* Use `HIP_VISIBLE_DEVICES=6`. We are on a shared node, so following this is critical!
* Our backward kernel should be compatible with the forward one, so must mirror what's necessary to mirror.
* Please do not add any nothing of "Phase" in your code comment. The "Phases" are for development only and keeping track on where were are, but are not of interests to any other person.
* If other documents are contradicting the constraints listed here, consider this document to superseed them.
* Writing optimization on kernel is a bit like searching a vast space of possibilities. We can see it as a journey, from one state to another. Sometimes we are standing on a stepping stone and move to another stepping stone which leads to regression. The question we should always ask ourselves is if this regression/failure is a "hard regression" (i.e. not on the path to the ultimate solution) or "soft regression" (*may* be on the path to the ultimate solution). Sometimes we have to do a go through a regression(s) to enable further improvements. It's obviously not always true. The difficulty is to distinguish between hard vs soft regression. If we don't view the solution strategy we might search the space in a very greedy way, and miss a lot of opportunities. So, please, *always* ask yourself the question about the stepping stone your are sitting in, and follow this discipline. Sometimes a bit of exploration is a sound investment.
* We are on limited resource server. If you need to write temporary data please do so in /data/tmp_hstu
