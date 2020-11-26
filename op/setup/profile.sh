#!/bin/bash

# this script must be run at project root by calling "op/setup/profile.sh"

# make sure env var $DGD is ready
PATH_PROFILE="$HOME/.bash_profile"
PATH_PROJECT=$(pwd)
EXPORT_LINE=$(grep "export DGD=" "$PATH_PROFILE")

# export env var
export DGD="$PATH_PROJECT"

# install env var in bash profile for subsequent shell session
if [ -z "$EXPORT_LINE" ]; then
  { echo ""; echo "# deepgaau-detector"; echo "export DGD=\"$PATH_PROJECT\""; } >> "$PATH_PROFILE"
  echo "env var \$DGD is set"
else
  echo "env var \$DGD has already been there"
fi
