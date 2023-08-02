function solver = Tests(mesh)
    solver = quadratic_case(mesh);
end
function tensor = get_tensor(nvols, value)
    tensor = (ones(3, 3, nvols) * value) .* eye(3);
end
function tensor = get_random_tensor(nvols)
    tensor = rand(3, 3, nvols) .* eye(3);
end 

function solver = linear_case(mesh)
    function d = dirichlet(x, y, z)
        d = x + y + z;
        return
    end
    function n = neumann(x, y, z)
        n = zeros(1, length(x));
    end
    function s = source(x, y, z)
        s = zeros(1, length(x));
    end
    permeability = get_tensor(mesh.nvols, 1.);
    xv = mesh.volumes_centers(:, 1);
    yv = mesh.volumes_centers(:, 2);
    zv = mesh.volumes_centers(:, 3);
    an = dirichlet(xv, yv, zv);
    keys = ["check", "name", "mesh", "permeability", "dirichlet", "neumann", "analytical", "source"];
    vals = {true, "linear_case", mesh, permeability, @dirichlet, @neumann, an, @source};
    args = dictionary(keys, vals);
    solver = Solver();
    solver = Solver();
    solver = solver.solve_TPFA(args);
end

function solver = quadratic_case(mesh)
    function d = dirichlet(x, y, z)
        d = x.^2 + y.^2 + z.^2;
    end
    function n = neumann(x, y, z)
        n = zeros(1, length(x));
    end
    function s = source(x, y, z)
        s = ones(1, length(x)) * -6;
    end
    permeability = get_tensor(mesh.nvols, 1.);
    xv = mesh.volumes_centers(:, 1);
    yv = mesh.volumes_centers(:, 2);
    zv = mesh.volumes_centers(:, 3);
    an = dirichlet(xv, yv, zv);
    keys = ["check", "name", "mesh", "permeability", "dirichlet", "neumann", "analytical", "source"];
    vals = {true, "quadratic_case", mesh, permeability, @dirichlet, @neumann, an, @source};
    args = dictionary(keys, vals);
    solver = Solver();
    solver = solver.solve_TPFA(args);
end

function solver = extra_case1(mesh)
    function d = dirichlet(x, y, z)
        d = sin(x) + cos(y) + exp(z);
    end
    function n = neumann(x, y, z)
        n = zeros(1, length(x));
    end
    function s = source(x, y, z)
        s = sin(x) + cos(y) - exp(z);
    end
    permeability = get_tensor(mesh.nvols, 1.);
    xv = mesh.volumes_centers(:, 1);
    yv = mesh.volumes_centers(:, 2);
    zv = mesh.volumes_centers(:, 3);
    an = dirichlet(xv, yv, zv);
    keys = ["check", "name", "mesh", "permeability", "dirichlet", "neumann", "analytical", "source"];
    vals = {true, "extra_case1", mesh, permeability, @dirichlet, @neumann, an, @source};
    args = dictionary(keys, vals);
    solver = Solver();
    solver = solver.solve_TPFA(args);
end

function solver = extra_case2(mesh)
    function d = dirichlet(x, y, z)
        d = (x + 1) .* log(1 + x) + 1 ./ (y + 1) + z.^2;
    end
    function n = neumann(x, y, z)
        n = zeros(1, length(x));
    end
    function s = source(x, y, z)
        s = -2 - (1 ./ (1 + x)) - (2 ./ ((1 + y) .^ 3));
    end
    permeability = get_tensor(mesh.nvols, 1.);
    xv = mesh.volumes_centers(:, 1);
    yv = mesh.volumes_centers(:, 2);
    zv = mesh.volumes_centers(:, 3);
    an = dirichlet(xv, yv, zv);
    keys = ["check", "name", "mesh", "permeability", "dirichlet", "neumann", "analytical", "source"];
    vals = {true, "extra_case2", mesh, permeability, @dirichlet, @neumann, an, @source};
    args = dictionary(keys, vals);
    solver = Solver();
    solver = solver.solve_TPFA(args);
end

function solver = qfive_spot(mesh, box_dimensions)
    Lx = box_dimensions(1);
    Ly = box_dimensions(2);
    Lz = box_dimensions(3);
    d1 = 10000.;
    d2 = 1.;
    k1 = 100.;
    k2 = 100.;
    vq = 0.;
    function d = dirichlet(x, y, z)
        p1 = 0.1;
        p2 = 0.1;
        l1, c1 = Lx / 8, Ly / 8;
        l2, c2 = Lx / 8, Ly / 8;
        dist0 = (x >= 0) & (x <= l1) & (y >= 0) & (y <= c1);
        dist1 = (x >= Lx - l2) & (x <= Lx) & (y >= Ly - c2) & (y <= Ly);
        v1 = find(dist0 == true);
        v2 = find(dist1 == true);
        
        d = zeros(1, length(x)) * NaN;
        d(v1) = d1;
        d(v2) = d2;

    end
    function a_p = analytical(x, y, z)
        ref = load("vtks/QFiveSpotRef.mat");

        xr  = ref(:, 1);
        yr  = ref(:, 2);
        pr  = ref(:, 4);

        n   = round(sqrt(length(x)));

        dx  = Lx / n;
        dy  = Ly / n;

        i = ceil(yr ./ dy);
        j = ceil(xr ./ dx);

        idx = (i - 1) .* n + j;
        a_p = zeros(1, n * n);
        freq = zeros(1, n * n);
        
        mx = size(idx);
        for i = 1:mx(2)
            a_p(idx(i)) = a_p(idx(i)) + pr(i);
            freq(idx(i)) = freq(idx(i)) + 1;
        end

        m = freq(1);
        a_p = a_p ./ m;

    end
    function n = neumann(x, y, z)
        n = zeros(1, length(x));
    end
    function s = source(x, y, z)
        s = ones(1, length(x)) * vq;
    end
    permeability = get_tensor(mesh.nvols, 1.);
    xv = mesh.volumes_centers(:, 1);
    yv = mesh.volumes_centers(:, 2);
    zv = mesh.volumes_centers(:, 3);
    an = dirichlet(xv, yv, zv);
    keys = ["check", "name", "mesh", "permeability", "dirichlet", "neumann", "analytical", "source"];
    vals = {true, "qfive_spot", mesh, permeability, @dirichlet, @neumann, an, @source};
    args = dictionary(keys, vals);
    solver = Solver();
    solver = solver.solve_TPFA(args);
end