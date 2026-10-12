%% SPM12 正式一阶分析：仅使用前 4 runs = 4 SPM sessions
%
% 数据结构：
%   每名被试原有 8 个 4-D NIfTI：0006--0013，每个 130 volumes。
%   每名被试有一个合并头动文件：headmove01--headmove100，1040 x 6。
%
% 统计模型：
%   1. 暂时仅使用前 4 个 run（0006--0009），后 4 个 run 不进入模型；
%   2. 每个保留的 run 建立为一个独立 SPM session；
%   3. 每个 session 使用对应的 130 行头动参数；
%   4. Simple、Complex、Control 显式建模，Rest 作为隐式基线；
%   5. 每个 session 独立高通滤波、常数项和时间自相关结构；
%   6. contrasts 使用 replsc，得到跨 4 个 session 的平均效应。

clear;
clc;

%% --------------------------- 用户设置 -------------------------------
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));

fmri_root = getenv('FMRI_VIDEO_FMRI_ROOT');
if isempty(fmri_root)
    fmri_root = fullfile(project_root, 'data', 'raw', 'fmrinii');
end

motion_root = getenv('FMRI_VIDEO_MOTION_ROOT');
if isempty(motion_root)
    motion_root = fullfile(project_root, 'data', 'raw', 'headmove');
end

base_path = getenv('FMRI_VIDEO_FIRST_LEVEL_OUTPUT');
if isempty(base_path)
    base_path = fullfile(project_root, 'results', ...
        'spm_first_level_4sessions');
end

excluded_subjects = [5, 7, 16, 31, 34, 38, 53, 62, 66, 67, 68, 73, 83];
selected_subjects = setdiff(1:100, excluded_subjects);

all_sessions = {'0006', '0007', '0008', '0009', ...
                '0010', '0011', '0012', '0013'};
sessions = all_sessions(1:4);
num_volumes = 130;
image_prefix = 'swraSMS_BOLD_2MM_REST_';

% 标准刚体头动参数应为 6 列。如果你的文件是 Friston-24，请改为 24，
% 并确认导数是在每个 run 内分别计算后再纵向合并的。
expected_motion_columns = 6;

% 完整 onset 沿用原脚本的拼接坐标（0--1039）。当前只保留前 4 个
% run，因此自动筛选 onset < 4*130，即全局坐标 0--519 内的事件。
condition_names = {'Simple', 'Complex', 'Control'};
all_global_onsets = {
    [10, 220, 310, 480, 570, 660, 830, 1000], ... % Simple
    [50, 140, 350, 400, 610, 700, 870, 920],  ... % Complex
    [90, 180, 270, 440, 530, 740, 790, 960]   ... % Control
};
selected_scan_count = numel(sessions) * num_volumes;
global_onsets = cellfun( ...
    @(onsets) onsets(onsets < selected_scan_count), ...
    all_global_onsets, 'UniformOutput', false);
condition_durations = [30, 30, 30];

continue_after_error = true;
run_results_report = true;

%% --------------------------- 初始化 SPM -----------------------------
if isempty(which('spm'))
    error('SPM12 不在 MATLAB 路径中，请先使用 addpath 添加 SPM12。');
end

spm('defaults', 'FMRI');
spm_jobman('initcfg');

failed_subjects = {};

%% --------------------------- 逐被试运行 -----------------------------
for subject_number = selected_subjects
    id = sprintf('%d', subject_number);
    fprintf('\n========== Subject %s ==========\n', id);

    try
        output_dir = fullfile(base_path, id);
        if ~exist(output_dir, 'dir')
            [success, message] = mkdir(output_dir);
            if ~success
                error('无法创建输出目录 %s：%s', output_dir, message);
            end
        end

        % 防止新的 4-session 模型与旧模型结果混在同一目录中。
        spm_mat = fullfile(output_dir, 'SPM.mat');
        if exist(spm_mat, 'file')
            error(['输出目录中已存在 SPM.mat：%s\n' ...
                   '请先移动旧结果，或为本次分析使用新的输出目录。'], ...
                  spm_mat);
        end

        matlabbatch = create_first_level_job( ...
            id, fmri_root, motion_root, output_dir, sessions, ...
            numel(all_sessions), num_volumes, image_prefix, ...
            expected_motion_columns, ...
            condition_names, global_onsets, condition_durations, ...
            run_results_report);

        spm_jobman('run', matlabbatch);
        fprintf('Finished first-level analysis for subject %s.\n', id);

    catch ME
        failed_subjects{end + 1} = id; %#ok<SAGROW>
        fprintf(2, '\nSubject %s failed:\n%s\n', id, ...
                getReport(ME, 'extended', 'hyperlinks', 'off'));

        if ~continue_after_error
            rethrow(ME);
        end
    end
end

fprintf('\n================ Summary ================\n');
fprintf('计划处理：%d 名被试；排除：%s\n', ...
        numel(selected_subjects), num2str(excluded_subjects));
if isempty(failed_subjects)
    fprintf('全部被试处理完成。\n');
else
    fprintf(2, '运行失败的被试（%d）：%s\n', ...
            numel(failed_subjects), strjoin(failed_subjects, ', '));
end


function matlabbatch = create_first_level_job( ...
        id, fmri_root, motion_root, output_dir, sessions, ...
        total_motion_sessions, num_volumes, image_prefix, ...
        expected_motion_columns, condition_names, global_onsets, ...
        condition_durations, run_results_report)
% 为一名被试建立并校验 4-session 一阶模型。

    fmri_dir = fullfile(fmri_root, id);
    if ~exist(fmri_dir, 'dir')
        error('被试 %s 的功能像目录不存在：%s', id, fmri_dir);
    end

    num_sessions = numel(sessions);
    num_conditions = numel(condition_names);

    if numel(global_onsets) ~= num_conditions || ...
            numel(condition_durations) ~= num_conditions
        error('condition_names、global_onsets 和 condition_durations 数量不一致。');
    end

    local_onsets = split_and_validate_onsets( ...
        global_onsets, condition_durations, num_sessions, num_volumes);

    motion_files = prepare_motion_files( ...
        id, motion_root, output_dir, sessions, num_volumes, ...
        total_motion_sessions, expected_motion_columns);

    matlabbatch = cell(1, 3 + double(run_results_report));
    matlabbatch{1}.spm.stats.fmri_spec.dir = {output_dir};
    matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'scans';
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 2;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 31;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 3;

    for session_index = 1:num_sessions
        session_id = sessions{session_index};
        image_file = fullfile( ...
            fmri_dir, sprintf('%s%s.nii', image_prefix, session_id));

        scans = make_and_validate_scan_list(image_file, num_volumes);
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_index).scans = scans;

        for condition_index = 1:num_conditions
            onsets = local_onsets{condition_index, session_index};
            if isempty(onsets)
                error('Session %s 中没有条件 %s。', ...
                      session_id, condition_names{condition_index});
            end

            condition = struct();
            condition.name = condition_names{condition_index};
            condition.onset = onsets(:);
            condition.duration = repmat( ...
                condition_durations(condition_index), numel(onsets), 1);
            condition.tmod = 0;
            condition.pmod = struct('name', {}, 'param', {}, 'poly', {});
            condition.orth = 0;

            matlabbatch{1}.spm.stats.fmri_spec.sess(session_index) ...
                .cond(condition_index) = condition;
        end

        matlabbatch{1}.spm.stats.fmri_spec.sess(session_index).multi = {''};
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_index).regress = ...
            struct('name', {}, 'val', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_index).multi_reg = ...
            {motion_files{session_index}};
        matlabbatch{1}.spm.stats.fmri_spec.sess(session_index).hpf = 128;
    end

    matlabbatch{1}.spm.stats.fmri_spec.fact = ...
        struct('name', {}, 'levels', {});
    matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0, 0];
    matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
    matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
    matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8;
    matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
    matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

    %% 模型估计
    matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep( ...
        'fMRI model specification: SPM.mat File', ...
        substruct('.','val', '{}',{1}, '.','val', '{}',{1}, ...
                  '.','val', '{}',{1}), ...
        substruct('.','spmmat'));
    matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

    %% Contrasts：Rest 为隐式基线
    matlabbatch{3}.spm.stats.con.spmmat(1) = cfg_dep( ...
        'Model estimation: SPM.mat File', ...
        substruct('.','val', '{}',{2}, '.','val', '{}',{1}, ...
                  '.','val', '{}',{1}), ...
        substruct('.','spmmat'));

    matlabbatch{3}.spm.stats.con.consess{1}.fcon.name = ...
        'AnyConditionVsRest';
    matlabbatch{3}.spm.stats.con.consess{1}.fcon.weights = eye(3);
    matlabbatch{3}.spm.stats.con.consess{1}.fcon.sessrep = 'replsc';

    contrast_names = {'SimActive', 'CompActive', 'ConActive', ...
                      'SimComp', 'SimCon', 'CompCon'};
    contrast_weights = {
        [1,  0,  0], ...
        [0,  1,  0], ...
        [0,  0,  1], ...
        [1, -1,  0], ...
        [1,  0, -1], ...
        [0,  1, -1]  ...
    };

    for contrast_index = 1:numel(contrast_names)
        batch_index = contrast_index + 1;
        matlabbatch{3}.spm.stats.con.consess{batch_index}.tcon.name = ...
            contrast_names{contrast_index};
        matlabbatch{3}.spm.stats.con.consess{batch_index}.tcon.weights = ...
            contrast_weights{contrast_index};
        matlabbatch{3}.spm.stats.con.consess{batch_index}.tcon.sessrep = ...
            'replsc';
    end
    matlabbatch{3}.spm.stats.con.delete = 0;

    %% 可选结果报告
    if run_results_report
        matlabbatch{4}.spm.stats.results.spmmat(1) = cfg_dep( ...
            'Contrast Manager: SPM.mat File', ...
            substruct('.','val', '{}',{3}, '.','val', '{}',{1}, ...
                      '.','val', '{}',{1}), ...
            substruct('.','spmmat'));
        matlabbatch{4}.spm.stats.results.conspec.titlestr = '';
        matlabbatch{4}.spm.stats.results.conspec.contrasts = Inf;
        matlabbatch{4}.spm.stats.results.conspec.threshdesc = 'FWE';
        matlabbatch{4}.spm.stats.results.conspec.thresh = 0.05;
        matlabbatch{4}.spm.stats.results.conspec.extent = 0;
        matlabbatch{4}.spm.stats.results.conspec.conjunction = 1;
        matlabbatch{4}.spm.stats.results.conspec.mask.none = 1;
        matlabbatch{4}.spm.stats.results.units = 1;
        matlabbatch{4}.spm.stats.results.export{1}.ps = true;
    end
end


function local_onsets = split_and_validate_onsets( ...
        global_onsets, durations, num_sessions, num_volumes)
% 将拼接坐标的 onset 转换为每个 run 内的 onset。

    num_conditions = numel(global_onsets);
    total_volumes = num_sessions * num_volumes;
    local_onsets = cell(num_conditions, num_sessions);

    for condition_index = 1:num_conditions
        onsets = global_onsets{condition_index};

        if ~isnumeric(onsets) || any(~isfinite(onsets(:)))
            error('条件 %d 的 onset 必须是有限数值。', condition_index);
        end
        if any(onsets < 0) || any(onsets >= total_volumes)
            error('条件 %d 中存在超出 0--%d 的 onset。', ...
                  condition_index, total_volumes - 1);
        end
        if ~isscalar(durations(condition_index)) || ...
                ~isfinite(durations(condition_index)) || ...
                durations(condition_index) < 0
            error('条件 %d 的 duration 无效。', condition_index);
        end

        for session_index = 1:num_sessions
            session_start = (session_index - 1) * num_volumes;
            in_session = onsets >= session_start & ...
                         onsets < session_start + num_volumes;
            current_onsets = onsets(in_session) - session_start;

            if any(current_onsets + durations(condition_index) > num_volumes)
                error('条件 %d 的任务块跨越了 session %d 的边界。', ...
                      condition_index, session_index);
            end
            local_onsets{condition_index, session_index} = current_onsets(:);
        end
    end
end


function scans = make_and_validate_scan_list(image_file, expected_volumes)
% 检查 4-D NIfTI，并生成 SPM 所需的 volume 路径。

    if ~exist(image_file, 'file')
        error('功能像不存在：%s', image_file);
    end

    try
        headers = spm_vol(image_file);
    catch ME
        error('SPM 无法读取 %s：%s', image_file, ME.message);
    end

    if numel(headers) ~= expected_volumes
        error('%s 包含 %d 个 volumes，预期为 %d。', ...
              image_file, numel(headers), expected_volumes);
    end

    scans = cell(expected_volumes, 1);
    for volume_index = 1:expected_volumes
        scans{volume_index} = sprintf('%s,%d', image_file, volume_index);
    end
end


function motion_files = prepare_motion_files( ...
        id, motion_root, output_dir, sessions, num_volumes, ...
        total_motion_sessions, expected_columns)
% 校验 1040 行合并头动文件，仅提取前 4 个 run 的头动参数。

    if ~exist(motion_root, 'dir')
        error('头动参数目录不存在：%s', motion_root);
    end

    num_sessions = numel(sessions);
    headmove_file = find_headmove_file(motion_root, id);
    combined_motion = load_motion_matrix(headmove_file);

    expected_rows = total_motion_sessions * num_volumes;
    if size(combined_motion, 1) ~= expected_rows
        error('被试 %s 的头动文件有 %d 行，预期为 %d。', ...
              id, size(combined_motion, 1), expected_rows);
    end
    if size(combined_motion, 2) ~= expected_columns
        error('被试 %s 的头动文件有 %d 列，预期为 %d。', ...
              id, size(combined_motion, 2), expected_columns);
    end
    if any(~isfinite(combined_motion(:)))
        error('被试 %s 的头动文件包含 NaN 或 Inf。', id);
    end

    split_dir = fullfile(output_dir, 'motion_by_session');
    if ~exist(split_dir, 'dir')
        [success, message] = mkdir(split_dir);
        if ~success
            error('无法创建头动参数目录 %s：%s', split_dir, message);
        end
    end

    motion_files = cell(num_sessions, 1);
    for session_index = 1:num_sessions
        first_row = (session_index - 1) * num_volumes + 1;
        last_row = session_index * num_volumes;
        motion_files{session_index} = fullfile( ...
            split_dir, sprintf('headmove_%s.txt', sessions{session_index}));

        dlmwrite(motion_files{session_index}, ...
                 combined_motion(first_row:last_row, :), ...
                 'delimiter', '\t', 'precision', '%.10g');
    end

    fprintf('被试 %s：已提取前 %d 个 run 的头动参数到 %s\n', ...
            id, num_sessions, split_dir);
end


function headmove_file = find_headmove_file(motion_root, id)
% 同时兼容 headmove01 和 headmove01.txt。

    subject_number = str2double(id);
    motion_stem = sprintf('headmove%02d', subject_number);
    candidates = {
        fullfile(motion_root, motion_stem), ...
        fullfile(motion_root, [motion_stem '.txt']) ...
    };
    exists_mask = cellfun(@(file) exist(file, 'file') == 2, candidates);

    if ~any(exists_mask)
        error('被试 %s 的头动文件不存在。尝试过：\n%s', ...
              id, strjoin(candidates, '\n'));
    end
    if sum(exists_mask) > 1
        error('被试 %s 同时存在两个候选头动文件：\n%s', ...
              id, strjoin(candidates(exists_mask), '\n'));
    end

    headmove_file = candidates{find(exists_mask, 1)};
end


function motion = load_motion_matrix(file_name)
% 读取纯数值头动文件，无扩展名时提供 ASCII 兼容读取。

    try
        motion = spm_load(file_name);
    catch spm_error
        try
            motion = load(file_name, '-ascii');
        catch ascii_error
            error(['无法读取头动文件 %s。\nSPM 错误：%s\n' ...
                   'ASCII 错误：%s'], ...
                  file_name, spm_error.message, ascii_error.message);
        end
    end

    if ~isnumeric(motion) || ndims(motion) ~= 2 || isempty(motion)
        error('头动文件不是非空数值矩阵：%s', file_name);
    end
end
