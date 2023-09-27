classdef Problem
    
    properties
        name
        handle
        meshfiles
        nvols_arr
        avg_error_arr
        avg_time_arr      
        avg_memory_arr      
    end

    methods
        function problem = init_problem(problem, name, handle, meshfiles)
            problem.name = name;
            problem.handle = handle;
            problem.meshfiles = meshfiles;
            problem.nvols_arr = zeros(1, length(meshfiles));
            problem.avg_error_arr = zeros(1, length(meshfiles));
            basic = dictionary();
            ks = [  "Pre-Processing*", ...
                    "TPFA System Preparation", ...
                    "TPFA Boundary Conditions", ...
                    "TPFA Solver", ...
                    "Post-Processing*"  ];
            basic(ks) = 0;
            problem.avg_time_arr = {};
            for i = 1:length(meshfiles)
                problem.avg_time_arr{end+1} = basic;
            end
            problem.avg_memory_arr = zeros(1, length(meshfiles));
        end
        function problem = add_to_problem(problem, i, j, nvols, error, time, memory)
            if i > length(problem.meshfiles)
                error('Index out of bounds');
            end
            problem.nvols_arr(i) = nvols;
            problem.avg_error_arr(i) = problem.avg_error_arr(i) * (j-1)/j + error/j;
            for key = time.keys()
                problem.avg_time_arr{i}(key) = problem.avg_time_arr{i}(key) * (j-1)/j + time(key)/j;
            end
            problem.avg_memory_arr(i) = problem.avg_memory_arr(i) * (j-1)/j + memory/j;
        end
    end
end