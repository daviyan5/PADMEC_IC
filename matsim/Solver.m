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
        d_values         % Values of Dirichlet boundary conditions
        d_volumes        % Volumes with Dirichlet boundary conditions

        n_faces          % Faces with Neumann boundary conditions
        n_values         % Values of Neumann boundary conditions
        n_volumes        % Volumes with Neumann boundary conditions

        bench_info       % Object with times and memories
        vtk_filename     % Name of the vtk file to be written
        error            % Error of the solution
    end

    methods
        function solver = solve_TPFA(solver, args)
            solver.times = dictionary;
            solver.check = args{"check"};
            solver.name  = args{"name"};

            step_name = 'Pre-Processing*';
            tic
                solver.mesh = args{"mesh"};
                solver.permeability = args{"permeability"};
            toc
            solver.times(step_name) = toc;
            
            step_name = 'TPFA System Preparation';
            tic
                solver = solver.assemble_faces_transmissibilities();
                solver = solver.assemble_TPFA_matrix(args{"source"});
            toc
            solver.times(step_name) = toc;
            
            step_name = 'TPFA Boundary Conditions';
            tic
                solver = solver.set_dirichlet_boundary_conditions(args{"dirichlet"});
                solver = solver.set_neumann_boundary_conditions(args{"neumann"});
            toc
            solver.times(step_name) = toc;

            step_name = 'TPFA Solver';
            tic
                solver = solver.solve_TPFA_system();
            toc
            solver.times(step_name) = toc;

            step_name = 'Post-Processing*';
            tic
                solver.analytical_p = args{"analytical"};
                solver.numerical_p  = solver.p_TPFA;
                solver.post_processing();
            toc
            solver.times(step_name) = toc;
            return;
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
            

        end
        function solver = set_neumann_boundary_conditions(solver, neumann)
        end
        function solver = solve_TPFA_system(solver)
        end
        function solver = post_processing(solver)
        end
    end
end