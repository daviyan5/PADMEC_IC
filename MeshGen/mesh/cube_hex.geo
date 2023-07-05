DefineConstant[
  N = {10, Name "Input/1Points "}
];
Lx=6;
hx=Lx/N;
Ly=4;
hy=Ly/N;
Lz=1;
hz=Lz/N;
h=Exp(Log(hx*hy*hz)/3);
// 0D
Point (1) = {0, 0, 0, h};
Point (2) = {Lx, 0, 0, h};
Point (3) = {Lx, Ly, 0, h};
Point (4) = {0, Ly, 0, h};
Point (5) = {0, 0, Lz, h};
Point (6) = {Lx, 0, Lz, h};
Point (7) = {Lx, Ly, Lz, h};
Point (8) = {0, Ly, Lz, h};

// 1D
Line (1)  = {1, 2};
Line (2)  = {2, 3};
Line (3)  = {3, 4};
Line (4)  = {4, 1};
Line (5)  = {5, 6};
Line (6)  = {6, 7};
Line (7)  = {7, 8};
Line (8)  = {8, 5};
Line (9)  = {1, 5};
Line (10) = {2, 6};
Line (11) = {3, 7};
Line (12) = {4, 8};

// 2D
Line Loop(13) = {3, 12, -7, -11};
Line Loop(15) = {4, 9, -8, -12};
Line Loop(17) = {5, -10, -1, 9};
Line Loop(19) = {3, 4, 1, 2};
Line Loop(21) = {11, -6, -10, 2};
Line Loop(23) = {7, 8, 5, 6};

Plane Surface(1) = {15};
Plane Surface(2) = {21};
Plane Surface(3) = {17};
Plane Surface(4) = {13};
Plane Surface(5) = {19};
Plane Surface(6) = {23};


// Transfinite for 2D
// Modificar aqui o número de subdivisões em cada direção.
// No final, será gerada uma malha (n-1) x (n-1) x (n-1), onde
// n é o número especificado abaixo.
Transfinite Line {1, 2, 3, 4, 5, 6, 7, 8} = 2;
Transfinite Line {9, 10, 11, 12} = 2;
Transfinite Surface "*";
Recombine Surface "*";

// 3D
Surface Loop(25) = {4, 5, 1, 3, 6, 2};
Volume(1) = {25};
Transfinite Volume "*";
Recombine Volume "*";


