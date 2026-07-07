# FlyDSL HSTU Backward Kernel Dev - General Notes

This document records all general information on where to find common references and the constraints you must follow.

#### Info

* We are running here on a MI350. 
* Our main repo is Aiter (that you can find at `/workspaces/git/meta/aiter`)
* Our colleague repo, Aiter fork, can be found at `/workspaces/git/meta/robin_aiter/`. He has implemented the FlyDSL HSTU forward kernel.
* Our theory document is at `2026-07-07_HSTU_theory.md`
* Our working python environment is at `/workspaces/git/meta/aiter/flydsl_venv`

#### Constraints

* Use `HIP_VISIBLE_DEVICES=6`. We are on a share node, so following this is critical!
* Our backward kernel should be compatible with the forward one, so must mirror what's necessary to mirror.
* Please do not add any nothing of "Phase" in your code comment. The "Phases" are for development only and keeping track on where were are, but are not of interests to any other person.
