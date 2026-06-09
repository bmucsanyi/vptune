# Blockers

- SPEC.md "Package Layout" (SPEC.md:2296-2348) pins the exact module list, which does not include the runtime owner modules now being introduced (`runtime_values.py`; further owner modules follow as runtime.py is converted). The final goal of the referenced session directs the conversion ("package layout is a consequence of ownership"), so the layout section needs a spec amendment once the conversion settles. Decision needed: amend SPEC.md Package Layout to the final module list, or fold the owner modules back.
