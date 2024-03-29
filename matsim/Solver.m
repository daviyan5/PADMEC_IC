classdef Solver
   
    properties
        check            % Flag for checking if solution is valid
        name             % Name of the solver
        times            % Dict for storing times

        mesh             % Mesh object
        permeability     % Permeability tensor flattened

        faces_normals    % Coordinates of normal vectors internal faces-wise
        faces_trans      % Transmissibility of each face

        h_L              % Distance from face_centers to left volume
        h_R              % Distance from face_centers to right volume

        At_TPFA          % Transmissibility matrix
        bt_TPFA          % Source term
        p_TPFA           % Pressure solution

        irel             % Relative error element-wise
        numerical_p      % Numerical pressure solution
        analytical_p     % Analytical pressure solution

        faces_with_bc    % Faces with boundary conditions
        volumes_with_bc  % Volumes with boundary conditions

        d_faces          % Faces with Dirichlet boundary conditions
        dvalues          % Values of Dirichlet boundary conditions
        dvolumes         % Volumes with Dirichlet boundary conditions

        n_faces          % Faces with Neumann boundary conditions
        nvalues          % Values of Neumann boundary conditions
        nvolumes         % Volumes with Neumann boundary conditions
        
        memory           % Memory estimation
        error            % Error of the solution
    end

    methods
        function solver = solve_TPFA(solver, args)
            solver.times = dictionary;
            solver.check = args{"check"};
            solver.name  = args{"name"};

            step_name = 'Pre-Processing*';
            tic;
                solver.mesh = args{"mesh"};
                solver.permeability = args{"permeability"};
            toc;
            solver.times(step_name) = toc;
            
            step_name = 'TPFA System Preparation';
            tic;
                solver = solver.assemble_faces_transmissibilities();
                solver = solver.assemble_TPFA_matrix(args{"source"});
            toc;
            solver.times(step_name) = toc;
            
            step_name = 'TPFA Boundary Conditions';
            tic;
                solver = solver.set_dirichlet_boundary_conditions(args{"dirichlet"});
                solver = solver.set_neumann_boundary_conditions(args{"neumann"});
            toc;
            solver.times(step_name) = toc;

            step_name = 'TPFA Solver';
            tic;
                solver = solver.solve_TPFA_system();
            toc;
            solver.times(step_name) = toc;

            step_name = 'Post-Processing*';
            tic;
                solver.analytical_p = args{"analytical"};
                solver.numerical_p  = solver.p_TPFA;
                num = ((solver.analytical_p - solver.p_TPFA) .^ 2) .* solver.mesh.volumes.';
                den = ((solver.analytical_p) .^ 2) .* solver.mesh.volumes.';
                solver.irel = sqrt(num ./ den);
                solver.error = sqrt(sum(num) / sum(den));
            toc;
            solver.times(step_name) = toc;
            solver.memory = whos("solver").bytes;
            solver = solver.post_processing();
        end
        function NvLvR = get_normals(solver)
            volumes_pairs   = solver.mesh.internal_faces_adj;
            internal_faces  = solver.mesh.internal_faces;
            faces_nodes     = solver.mesh.faces_connectivity(internal_faces, :);

            i = solver.mesh.nodes_centers(faces_nodes(:, 1), :);
            j = solver.mesh.nodes_centers(faces_nodes(:, 2), :);
            k = solver.mesh.nodes_centers(faces_nodes(:, 3), :);

            nvols_pairs = size(volumes_pairs, 1);
            nvols = size(volumes_pairs(1, :), 2);

            volumes_centers = reshape(solver.mesh.volumes_centers(volumes_pairs, :, :), nvols_pairs, nvols, 3);
            faces_centers   = solver.mesh.faces_centers(internal_faces, :);
            L = squeeze(volumes_centers(:, 1, :));
            vL = faces_centers - L;

            if nvols > 1
                R = squeeze(volumes_centers(:, 2, :));
                vR = faces_centers - R;
            else
                vR = NaN;
            end
            N = cross(i - j, k - j);
            NvLvR = {N, vL, vR};   
        end
        function solver = assemble_faces_transmissibilities(solver)
            NvLvR = solver.get_normals();
            N = NvLvR{1};
            vL = NvLvR{2};
            vR = NvLvR{3};
            solver.faces_normals = abs(N) ./ sqrt(sum(N.^2, 2));

            solver.h_L = abs(dot(solver.faces_normals, vL, 2));
            solver.h_R = abs(dot(solver.faces_normals, vR, 2));

            KL = solver.permeability(:, :, solver.mesh.internal_faces_adj(:, 1));
            KR = solver.permeability(:, :, solver.mesh.internal_faces_adj(:, 2));
            sz  = size(solver.faces_normals);

            KpL = squeeze(pagemtimes(KL, reshape(solver.faces_normals.', sz(2), 1, sz(1)))).';
            KnL = dot(solver.faces_normals, KpL, 2);

            KpR = squeeze(pagemtimes(KR, reshape(solver.faces_normals.', sz(2), 1, sz(1)))).';
            KnR = dot(solver.faces_normals, KpR, 2);

            Keq = (KnL .* KnR) ./ (KnL .* solver.h_R + KnR .* solver.h_L);
            solver.faces_trans = Keq.' .* solver.mesh.faces_areas(solver.mesh.internal_faces);
        end
        function solver = assemble_TPFA_matrix(solver, source)
            row_index_p = solver.mesh.internal_faces_adj(:, 1);
            col_index_p = solver.mesh.internal_faces_adj(:, 2);

            row_index = [row_index_p; col_index_p];
            col_index = [col_index_p; row_index_p];
            data      = [solver.faces_trans, solver.faces_trans] .* -1;

            solver.At_TPFA = sparse(row_index, col_index, data, solver.mesh.nvols, solver.mesh.nvols);

            xv = solver.mesh.volumes_centers(:, 1);
            yv = solver.mesh.volumes_centers(:, 2);
            zv = solver.mesh.volumes_centers(:, 3);

            solver.bt_TPFA = source(xv, yv, zv) .* solver.mesh.volumes;
        
            v = sum(solver.At_TPFA, 2) .* -1;
            solver.At_TPFA = spdiags(v, 0, solver.At_TPFA);
            if solver.check
                assert(all(sum(solver.At_TPFA, 2) <= 1e-13))
            end
        end
        function solver = set_dirichlet_boundary_conditions(solver, dirichlet)
            d_volumes = unique(solver.mesh.boundary_faces_adj);
            d_volumes = setdiff(d_volumes, solver.volumes_with_bc);
            xv = solver.mesh.volumes_centers(d_volumes, 1);
            yv = solver.mesh.volumes_centers(d_volumes, 2);
            zv = solver.mesh.volumes_centers(d_volumes, 3);

            d_values = dirichlet(xv, yv, zv);
            mask     = find(~isnan(d_values));
            d_values = d_values(mask);
            d_volumes = d_volumes(mask);
            solver.volumes_with_bc = union(solver.volumes_with_bc, d_volumes);
            assert(isempty(d_volumes) == 0)

            solver.bt_TPFA(d_volumes) = d_values;
            solver.At_TPFA(d_volumes, :) = 0;
            v = zeros(solver.mesh.nvols, 1);
            v(d_volumes) = 1;
            solver.At_TPFA = solver.At_TPFA + spdiags(v, 0, solver.mesh.nvols, solver.mesh.nvols);
            solver.At_TPFA = sparse(solver.At_TPFA);

            solver.dvolumes = d_volumes;
            solver.dvalues  = d_values;

        end
        function solver = set_neumann_boundary_conditions(solver, neumann)
            n_volumes = unique(solver.mesh.boundary_faces_adj);
            n_volumes = setdiff(n_volumes, solver.volumes_with_bc);
            xv = solver.mesh.volumes_centers(n_volumes, 1);
            yv = solver.mesh.volumes_centers(n_volumes, 2);
            zv = solver.mesh.volumes_centers(n_volumes, 3);

            n_values  = neumann(xv, yv, zv);
            mask      = find(~isnan(n_values));
            n_values  = n_values(mask);
            n_volumes = n_volumes(mask);
            solver.volumes_with_bc = union(solver.volumes_with_bc, n_volumes);
            if(isempty(n_volumes))
                return
            end

            solver.bt_TPFA(n_volumes) = solver.bt_TPFA(n_volumes) + n_values;
            
            solver.nvolumes = n_volumes;
            solver.nvalues  = n_values;
        end
        function solver = solve_TPFA_system(solver)
            solver.p_TPFA = solver.At_TPFA \ solver.bt_TPFA.';
            if solver.check
                assert(all(sum(solver.At_TPFA * solver.p_TPFA - solver.bt_TPFA.', 2) <= 1e-10))
            end
        end
        function solver = post_processing(solver)
            %nvols = solver.mesh.nvols;
            %error = solver.error;
            %times = solver.times;
            %memory = solver.memory;
            %save(sprintf("./vtks/solver_%d_%s.mat", solver.mesh.nvols, solver.name), "nvols", "error", "times", "memory");
        end
    end
end