#!/bin/bash
nbdev_build_lib
nbdev_build_docs --force_all TRUE
git add .
git commit -m 'autobuild'
git push


