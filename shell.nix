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
    # Demonstration notebooks
    jupyter_core notebook # Basic Jupyter notebook support
    pandas # For loading data
    matplotlib pygraphviz # For visualization
    # Performance improvement
    line_profiler
  ]);
in pkgs.mkShell {
  buildInputs = [
    mypython
    pkgs.ruff pkgs.isort pkgs.mypy
    pkgs.memray
  ];
  shellHook = ''
        alias pip="PIP_PREFIX='$(pwd)/_build/pip_packages' \pip"
        export PYTHONPATH="$(pwd)/_build/pip_packages/lib/python3.8/site-packages:$PYTHONPATH"
        unset SOURCE_DATE_EPOCH
  '';
}
