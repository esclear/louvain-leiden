let pkgs = import <nixpkgs> {};
mypython = pkgs.python3.withPackages (ps: with ps; [ networkx black flake8 pytest virtualenv pip pre-commit jupyter_core notebook]);
in pkgs.mkShell {
  buildInputs = [
    mypython
  ];
  shellHook = ''
        alias pip="PIP_PREFIX='$(pwd)/_build/pip_packages' \pip"
        export PYTHONPATH="$(pwd)/_build/pip_packages/lib/python3.8/site-packages:$PYTHONPATH"
        unset SOURCE_DATE_EPOCH
  '';
}
