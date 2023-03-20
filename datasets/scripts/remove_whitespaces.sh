#!/bin/bash
sed -E -i 's/[[:space:]]+/ /g' "$1"
echo "Done"
