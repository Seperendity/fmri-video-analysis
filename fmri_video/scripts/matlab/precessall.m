%-----------------------------------------------------------------------
% Job saved on 25-Feb-2025 15:08:33 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7771)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
%%
% 生成100个被试的ID列表
ID = cell(100, 1);
for i = 1:100
    ID{i} = sprintf('%d', i);
end

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));

fmri_root = getenv('FMRI_VIDEO_FMRI_ROOT');
if isempty(fmri_root)
    fmri_root = fullfile(project_root, 'data', 'raw', 'fmrinii');
end

t1_root = getenv('FMRI_VIDEO_T1_ROOT');
if isempty(t1_root)
    t1_root = fullfile(project_root, 'data', 'raw', 't1');
end

figure_root = getenv('FMRI_VIDEO_PREPROCESS_FIGURE_OUTPUT');
if isempty(figure_root)
    figure_root = fullfile(project_root, 'results', 'preprocess_figures');
end

% 循环处理指定被试
for i = 25:30
% for i = 2:length(ID)
    % 检查并创建输出目录
    output_dir = fullfile(figure_root, ID{i});
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % 设置工作目录为被试的输出目录（确保 .ps 文件保存到这里）
    cd(output_dir);

    % 生成当前被试的matlabbatch
    matlabbatch = preprocessing_job(ID{i}, fmri_root, t1_root);

    % 运行SPM预处理
    spm_jobman('run', matlabbatch);

    % 显式保存图形文件，加入被试 ID
    spm_figure('Create', 'Graphics', 'All');  % 捕获所有图形
    ps_filename = ['spm_' ID{i} '_2025Feb26.ps'];  % 自定义文件名，包含被试 ID
    spm_print(ps_filename);  % 保存到当前工作目录（output_dir）

    % 打印完成信息
    fprintf('Finished preprocessing for subject %s\n', ID{i});
end

function matlabbatch = preprocessing_job(id, fmri_root, t1_root)
    % 定义路径
    fmri_base_path = [fullfile(fmri_root, id) filesep];
    t1_base_path = [fullfile(t1_root, id) filesep];


    % 定义session文件名
    sessions = {'0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013'};
    num_volumes = 130; % 每个session的体视数

    % 第一步：Slice Timing
    matlabbatch{1}.spm.temporal.st.scans = cell(1, length(sessions));
    for s = 1:length(sessions)
        jsonsub = [fmri_base_path 'SMS_BOLD_2MM_REST_' sessions{s} '.json'];
        scan_list = cell(num_volumes, 1);
        for v = 1:num_volumes
            scan_list{v} = [fmri_base_path 'SMS_BOLD_2MM_REST_' sessions{s} '.nii,' num2str(v)];
            % disp(['当前文件路径：', scan_list{v}]);
        end
        matlabbatch{1}.spm.temporal.st.scans{s} = scan_list;
    end
    %%
    matlabbatch{1}.spm.temporal.st.nslices = 62;
    matlabbatch{1}.spm.temporal.st.tr = 2;
    matlabbatch{1}.spm.temporal.st.ta = 0;
    % read slice timing from json file of first run
    json = jsondecode(fileread(jsonsub));
    matlabbatch{1}.spm.temporal.st.so = json.SliceTiming;
    % disp(matlabbatch{1}.spm.temporal.st.so);
    % matlabbatch{1}.spm.temporal.st.so = [0.96 0 1.0225 0.065 1.0875 0.13 1.15 0.1925 1.215 0.2575 1.2775 0.32 1.3425 0.385 1.405 0.4475 1.47 0.5125 1.535 0.575 1.5975 0.64 1.6625 0.7025 1.725 0.7675 1.79 0.8325 1.8525 0.895 1.9175 0.96 0 1.0225 0.065 1.0875 0.13 1.15 0.1925 1.215 0.2575 1.2775 0.32 1.3425 0.385 1.405 0.4475 1.47 0.5125 1.535 0.575 1.5975 0.64 1.6625 0.7025 1.725 0.7675 1.79 0.8325 1.8525 0.895 1.9175];
    matlabbatch{1}.spm.temporal.st.refslice = 1.02;
    matlabbatch{1}.spm.temporal.st.prefix = 'a';
    matlabbatch{2}.spm.spatial.realign.estwrite.data{1}(1) = cfg_dep('Slice Timing: Slice Timing Corr. Images (Sess 1)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','files'));
    matlabbatch{2}.spm.spatial.realign.estwrite.data{2}(1) = cfg_dep('Slice Timing: Slice Timing Corr. Images (Sess 2)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{2}, '.','files'));
    matlabbatch{2}.spm.spatial.realign.estwrite.data{3}(1) = cfg_dep('Slice Timing: Slice Timing Corr. Images (Sess 3)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{3}, '.','files'));
    matlabbatch{2}.spm.spatial.realign.estwrite.data{4}(1) = cfg_dep('Slice Timing: Slice Timing Corr. Images (Sess 4)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{4}, '.','files'));
    matlabbatch{2}.spm.spatial.realign.estwrite.data{5}(1) = cfg_dep('Slice Timing: Slice Timing Corr. Images (Sess 5)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{5}, '.','files'));
    matlabbatch{2}.spm.spatial.realign.estwrite.data{6}(1) = cfg_dep('Slice Timing: Slice Timing Corr. Images (Sess 6)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{6}, '.','files'));
    matlabbatch{2}.spm.spatial.realign.estwrite.data{7}(1) = cfg_dep('Slice Timing: Slice Timing Corr. Images (Sess 7)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{7}, '.','files'));
    matlabbatch{2}.spm.spatial.realign.estwrite.data{8}(1) = cfg_dep('Slice Timing: Slice Timing Corr. Images (Sess 8)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{8}, '.','files'));
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.quality = 0.9;
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.sep = 4;
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.fwhm = 5;
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.rtm = 1;
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.interp = 2;
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.wrap = [0 0 0];
    matlabbatch{2}.spm.spatial.realign.estwrite.eoptions.weight = '';
    matlabbatch{2}.spm.spatial.realign.estwrite.roptions.which = [2 1];
    matlabbatch{2}.spm.spatial.realign.estwrite.roptions.interp = 4;
    matlabbatch{2}.spm.spatial.realign.estwrite.roptions.wrap = [0 0 0];
    matlabbatch{2}.spm.spatial.realign.estwrite.roptions.mask = 1;
    matlabbatch{2}.spm.spatial.realign.estwrite.roptions.prefix = 'r';
    matlabbatch{3}.spm.spatial.coreg.estimate.ref = {[t1_base_path 'T1.nii,1']};
    matlabbatch{3}.spm.spatial.coreg.estimate.source(1) = cfg_dep('Realign: Estimate & Reslice: Mean Image', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','rmean'));
    %%
    matlabbatch{3}.spm.spatial.coreg.estimate.other(1) = cfg_dep('Realign: Estimate & Reslice: Resliced Images (Sess 1)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{1}, '.','rfiles'));
    matlabbatch{3}.spm.spatial.coreg.estimate.other(2) = cfg_dep('Realign: Estimate & Reslice: Resliced Images (Sess 2)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{2}, '.','rfiles'));
    matlabbatch{3}.spm.spatial.coreg.estimate.other(3) = cfg_dep('Realign: Estimate & Reslice: Resliced Images (Sess 3)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{3}, '.','rfiles'));
    matlabbatch{3}.spm.spatial.coreg.estimate.other(4) = cfg_dep('Realign: Estimate & Reslice: Resliced Images (Sess 4)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{4}, '.','rfiles'));
    matlabbatch{3}.spm.spatial.coreg.estimate.other(5) = cfg_dep('Realign: Estimate & Reslice: Resliced Images (Sess 5)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{5}, '.','rfiles'));
    matlabbatch{3}.spm.spatial.coreg.estimate.other(6) = cfg_dep('Realign: Estimate & Reslice: Resliced Images (Sess 6)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{6}, '.','rfiles'));
    matlabbatch{3}.spm.spatial.coreg.estimate.other(7) = cfg_dep('Realign: Estimate & Reslice: Resliced Images (Sess 7)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{7}, '.','rfiles'));
    matlabbatch{3}.spm.spatial.coreg.estimate.other(8) = cfg_dep('Realign: Estimate & Reslice: Resliced Images (Sess 8)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{8}, '.','rfiles'));
    %%
    matlabbatch{3}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
    matlabbatch{3}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
    matlabbatch{3}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
    matlabbatch{3}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];
    matlabbatch{4}.spm.spatial.normalise.estwrite.subj.vol = {[t1_base_path 'T1.nii,1']};
    matlabbatch{4}.spm.spatial.normalise.estwrite.subj.resample(1) = cfg_dep('Realign: Estimate & Reslice: Resliced Images (Sess 1)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{1}, '.','rfiles'));
    matlabbatch{4}.spm.spatial.normalise.estwrite.subj.resample(2) = cfg_dep('Realign: Estimate & Reslice: Resliced Images (Sess 2)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{2}, '.','rfiles'));
    matlabbatch{4}.spm.spatial.normalise.estwrite.subj.resample(3) = cfg_dep('Realign: Estimate & Reslice: Resliced Images (Sess 3)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{3}, '.','rfiles'));
    matlabbatch{4}.spm.spatial.normalise.estwrite.subj.resample(4) = cfg_dep('Realign: Estimate & Reslice: Resliced Images (Sess 4)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{4}, '.','rfiles'));
    matlabbatch{4}.spm.spatial.normalise.estwrite.subj.resample(5) = cfg_dep('Realign: Estimate & Reslice: Resliced Images (Sess 5)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{5}, '.','rfiles'));
    matlabbatch{4}.spm.spatial.normalise.estwrite.subj.resample(6) = cfg_dep('Realign: Estimate & Reslice: Resliced Images (Sess 6)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{6}, '.','rfiles'));
    matlabbatch{4}.spm.spatial.normalise.estwrite.subj.resample(7) = cfg_dep('Realign: Estimate & Reslice: Resliced Images (Sess 7)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{7}, '.','rfiles'));
    matlabbatch{4}.spm.spatial.normalise.estwrite.subj.resample(8) = cfg_dep('Realign: Estimate & Reslice: Resliced Images (Sess 8)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{8}, '.','rfiles'));
    matlabbatch{4}.spm.spatial.normalise.estwrite.eoptions.biasreg = 0.0001;
    matlabbatch{4}.spm.spatial.normalise.estwrite.eoptions.biasfwhm = 60;
    matlabbatch{4}.spm.spatial.normalise.estwrite.eoptions.tpm = ...
        {fullfile(spm('Dir'), 'tpm', 'TPM.nii')};
    matlabbatch{4}.spm.spatial.normalise.estwrite.eoptions.affreg = 'eastern';
    matlabbatch{4}.spm.spatial.normalise.estwrite.eoptions.reg = [0 0.001 0.5 0.05 0.2];
    matlabbatch{4}.spm.spatial.normalise.estwrite.eoptions.fwhm = 0;
    matlabbatch{4}.spm.spatial.normalise.estwrite.eoptions.samp = 3;
    matlabbatch{4}.spm.spatial.normalise.estwrite.woptions.bb = [-90 -126 -72
                                                                 90 90 108];
    matlabbatch{4}.spm.spatial.normalise.estwrite.woptions.vox = [3 3 3];
    matlabbatch{4}.spm.spatial.normalise.estwrite.woptions.interp = 4;
    matlabbatch{4}.spm.spatial.normalise.estwrite.woptions.prefix = 'w';
    matlabbatch{5}.spm.spatial.smooth.data(1) = cfg_dep('Normalise: Estimate & Write: Normalised Images (Subj 1)', substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','files'));
    matlabbatch{5}.spm.spatial.smooth.fwhm = [8 8 8];
    matlabbatch{5}.spm.spatial.smooth.dtype = 0;
    matlabbatch{5}.spm.spatial.smooth.im = 0;
    matlabbatch{5}.spm.spatial.smooth.prefix = 's';
end
