#!/usr/bin/env bash
set -euo pipefail

install_root="${DBS_DIFFUSION_TOOLS_ROOT:-/home/devold/opt}"
mkdir -p "$install_root"
mkdir -p "$install_root/python-shim"
if [[ ! -e "$install_root/python-shim/python" ]]; then
  ln -s /usr/bin/python3 "$install_root/python-shim/python"
fi

clone_and_build() {
  local name="$1"
  local repository="$2"
  local tag="$3"
  local expected_commit="$4"
  local targets="${5:-}"
  local destination="$install_root/$name"

  if [[ -d "$destination/.git" ]]; then
    echo "Resuming verified source tree: $destination"
  elif [[ -e "$destination" ]]; then
    echo "Refusing non-repository path: $destination" >&2
    return 1
  else
    git clone --branch "$tag" --depth 1 "$repository" "$destination"
  fi

  local actual_commit
  actual_commit="$(git -C "$destination" rev-parse HEAD)"
  if [[ "$actual_commit" != "$expected_commit" ]]; then
    echo "Commit mismatch for $name: expected $expected_commit, got $actual_commit" >&2
    return 1
  fi

  local build_required=1
  if [[ -n "$targets" ]]; then
    build_required=0
    for target in $targets; do
      if [[ ! -x "$destination/$target" ]]; then
        build_required=1
        break
      fi
    done
  fi
  if [[ "$build_required" -eq 0 ]]; then
    echo "All requested binaries already exist; skipping build for $name."
    return 0
  fi

  (
    cd "$destination"
    python3 ./configure -nogui
    if [[ -n "$targets" ]]; then
      # Intentional word splitting: each target is one build argument.
      # shellcheck disable=SC2086
      NUMBER_OF_PROCESSORS="${DBS_BUILD_JOBS:-2}" python3 ./build $targets
    else
      NUMBER_OF_PROCESSORS="${DBS_BUILD_JOBS:-2}" python3 ./build
    fi
  )
}

clone_and_build \
  "mrtrix3-3.0.8" \
  "https://github.com/MRtrix3/mrtrix3.git" \
  "3.0.8" \
  "464173c5294e021b2d508e557085d90c09d642fd" \
  "bin/mrconvert bin/mrinfo bin/mrcalc bin/mrmath bin/mrstats bin/dwidenoise bin/mrdegibbs bin/dwi2mask bin/dwiextract bin/dwi2tensor bin/tensor2metric"

eigen_root="$install_root/eigen-3.3.9"
if [[ ! -d "$eigen_root/.git" ]]; then
  git clone --branch 3.3.9 --depth 1 \
    "https://gitlab.com/libeigen/eigen.git" "$eigen_root"
fi
eigen_commit="$(git -C "$eigen_root" rev-parse HEAD)"
if [[ "$eigen_commit" != "0fd6b4f71dd85b2009ee4d1aeb296e2c11fc9d68" ]]; then
  echo "Eigen 3.3.9 commit mismatch: $eigen_commit" >&2
  exit 1
fi
export CPLUS_INCLUDE_PATH="$eigen_root"

clone_and_build \
  "MRtrix3Tissue-5.2.9" \
  "https://github.com/3Tissue/MRtrix3Tissue.git" \
  "3Tissue_v5.2.9" \
  "f8ba0e3fa17cb3b6bb51bcc4939cde83243b8f4c" \
  "bin/dwi2fod bin/mtnormalise bin/mrconvert bin/mrinfo"

"$install_root/mrtrix3-3.0.8/bin/mrconvert" -version
"$install_root/MRtrix3Tissue-5.2.9/bin/ss3t_csd_beta1" -version
