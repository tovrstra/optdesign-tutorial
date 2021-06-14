#!/usr/bin/env bash

# Strip withspace form markdown files
for file in $(find . | grep 'md$'); do
  echo Cleaning ${file}
  sed -i -e $'s/\t/    /g' ${file}
  sed -i -e $'s/[ \t]\+$//' ${file}
  sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}' ${file}
done

for file in $(git ls-files | grep '\.py$'); do
  echo Cleaning ${file}
  autopep8 --max-line-length 100 --in-place ${file}
done

# Remove temporary files
rm -f */*.nbconvert.ipynb
rm -rf .ipynb_checkpoints
rm -rf .pytest_cache

# Strip output cells from notebooks
for file in $(git ls-files | grep '\.ipynb$'); do
  echo Cleaning ${file}
  jupyter nbconvert ${file} --to notebook  --inplace --nbformat 4 --ClearOutputPreprocessor.enabled=True
done

