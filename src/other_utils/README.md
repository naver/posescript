# Other potentially useful code

This directory presents some code that has been useful in different stages of the project (eg. for dataset construction), but does not contribute directly to the training or testing of the models proposed in the papers. In particular, this code is not necessarily cleaned nor well documented, and may trigger a bunch of errors / require some additional setup before running successfully (watch for "TODO" marks) -- apologies! Hopefully, this code can still prove useful for concrete understanding of some aspects of the project.

Included:
* pose mining (pose selection in PoseScript; also corresponding to poses "B" in PoseFix)
* pair mining (process to select pairs in PoseFix)

*Note:* for the reasons mentioned previously, this code is currently seccluded in this separate directory. Note that it requires some functions defined in the rest of the repository, so one may need to run `python setup.py develop` to import text2pose as a package first...