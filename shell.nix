let pkgs = import <nixpkgs> {};
mypython = pkgs.python3.withPackages (ps: with ps; [
    # For the library
    networkx
    # Testing
    pytest pytest-cov
    # Demonstration notebooks
    jupyter_core notebook # Basic Jupyter notebook support
    pandas # For loading data
    matplotlib # For visualization
  ]);
in pkgs.mkShell {
  buildInputs = [
    mypython
    # For static code analysis and formatting
    pkgs.ruff pkgs.mypy pkgs.black
  ];
  shellHook = ''
        alias pip="PIP_PREFIX='$(pwd)/_build/pip_packages' \pip"
        export PYTHONPATH="$(pwd)/_build/pip_packages/lib/python3.8/site-packages:$PYTHONPATH"
        unset SOURCE_DATE_EPOCH
  '';
}
