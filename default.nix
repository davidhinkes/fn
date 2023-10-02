{ }:
let pkgs = import <nixpkgs> {}; in

pkgs.mkShell {
  packages = [ pkgs.go pkgs.gnumake ];
}
