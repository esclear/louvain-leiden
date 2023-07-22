let pkgs = import <nixpkgs> {};
mypython = pkgs.python3.withPackages (ps: with ps; [
    # For the library
    networkx scipy
    # Development (code formatting)
    black flake8
    # Testing
    pytest pytest-cov
    # Package management
    virtualenv pip
    # Demonstration notebook
    jupyter_core notebook matplotlib
  ]);
in pkgs.mkShell {
  buildInputs = [
    mypython
    ruff
    pkgs.pre-commit
  ];
  shellHook = ''
        alias pip="PIP_PREFIX='$(pwd)/_build/pip_packages' \pip"
        export PYTHONPATH="$(pwd)/_build/pip_packages/lib/python3.8/site-packages:$PYTHONPATH"
        unset SOURCE_DATE_EPOCH
  '';
}
