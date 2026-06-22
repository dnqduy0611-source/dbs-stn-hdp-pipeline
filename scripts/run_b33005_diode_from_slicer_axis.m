function results = run_b33005_diode_from_slicer_axis(varargin)
% Run Medtronic B33005 DiODe using the Slicer-derived electrode axes.
%
% This wrapper is intentionally read-only with respect to Lead-DBS patient
% reconstruction files. It loads CT + Slicer-derived coordinates, calls
% ea_diode_medtronic for each side, and writes QC outputs only.
%
% Important convention:
%   Lead-DBS B33005 "head" is the K0 center at 1.65 mm from the distal
%   anchor, not the distal tip/anchor itself. This wrapper therefore uses
%   sub-PatientExample_B33005_axis_level_centers.csv K0 coordinates as head_mm.

p = inputParser;
p.addParameter('LeadDBSRoot', 'D:\dbs_workspace\tools\leaddbs_source\leaddbs-develop', @ischar);
p.addParameter('ElectrodesCsv', 'D:\dbs_workspace\dbs_stn_hdp_pipeline\patients\patient_example\derivatives\lead_dbs\sub-PatientExample_electrodefiducials_spaces.csv', @ischar);
p.addParameter('LevelsCsv', 'D:\dbs_workspace\dbs_stn_hdp_pipeline\patients\patient_example\derivatives\lead_dbs\sub-PatientExample_B33005_axis_level_centers.csv', @ischar);
p.addParameter('CTPath', 'D:\dbs_workspace\lead_dbs_work\PatientExampleLeadDBS6\derivatives\leaddbs\sub-PatientExample\coregistration\anat\sub-PatientExample_ses-postop_space-anchorNative_desc-preproc_CT.nii', @ischar);
p.addParameter('OutputDir', 'D:\dbs_workspace\dbs_stn_hdp_pipeline\patients\patient_example\derivatives\lead_dbs\diode_matlab_qc', @ischar);
p.addParameter('ElectrodeModel', 'Medtronic B33005', @ischar);
p.addParameter('Sides', [1 2], @isnumeric);
p.parse(varargin{:});
cfg = p.Results;

if ~isfolder(cfg.LeadDBSRoot)
    error('LeadDBSRoot does not exist: %s', cfg.LeadDBSRoot);
end
if ~isfile(cfg.ElectrodesCsv)
    error('ElectrodesCsv does not exist: %s', cfg.ElectrodesCsv);
end
if ~isfile(cfg.LevelsCsv)
    error('LevelsCsv does not exist: %s', cfg.LevelsCsv);
end
if ~isfile(cfg.CTPath)
    error('CTPath does not exist: %s', cfg.CTPath);
end
if ~isfolder(cfg.OutputDir)
    mkdir(cfg.OutputDir);
end

addpath(genpath(cfg.LeadDBSRoot));

logPath = fullfile(cfg.OutputDir, 'sub-PatientExample_B33005_diode_from_slicer_axis.log');
if isfile(logPath)
    delete(logPath);
end
diary(logPath);
cleanupObj = onCleanup(@() diary('off'));

fprintf('Run timestamp: %s\n', datestr(now, 31));
fprintf('MATLAB version: %s\n', version);
fprintf('LeadDBSRoot: %s\n', cfg.LeadDBSRoot);
fprintf('CTPath: %s\n', cfg.CTPath);
fprintf('ElectrodesCsv: %s\n', cfg.ElectrodesCsv);
fprintf('LevelsCsv: %s\n', cfg.LevelsCsv);
fprintf('OutputDir: %s\n', cfg.OutputDir);

options = struct();
options.elmodel = cfg.ElectrodeModel;
options = ea_resolve_elspec(options);
elspec = options.elspec;
fprintf('Electrode model: %s\n', elspec.elmodel);
fprintf('elspec.matfname: %s\n', elspec.matfname);
fprintf('lead_diameter: %.4f contact_length: %.4f contact_spacing: %.4f tip_length: %.4f markerpos: %.4f\n', ...
    elspec.lead_diameter, elspec.contact_length, elspec.contact_spacing, elspec.tip_length, elspec.markerpos);

global EA_DIODE_AUTOSELECT
EA_DIODE_AUTOSELECT = 'intensity';
autoSelectCleanup = onCleanup(@() clear('global', 'EA_DIODE_AUTOSELECT')); %#ok<NASGU>
fprintf('DiODe auto-select mode: %s\n', EA_DIODE_AUTOSELECT);

ct = ea_load_nii(cfg.CTPath);
tmat_vx2mm = ct.mat;

fidRows = readtable(cfg.ElectrodesCsv, 'TextType', 'string');
levelRows = readtable(cfg.LevelsCsv, 'TextType', 'string');

sideNames = {'right', 'left'};
sourceLabels = {{'E0','E1','E2','E3','E4','E5','E6','E7'}, {'E8','E9','E10','E11','E12','E13','E14','E15'}};

results = struct();
summaryRows = {};

for side = cfg.Sides
    if side < 1 || side > 2
        error('This wrapper currently supports side 1/right and 2/left only.');
    end
    sideName = sideNames{side};
    fprintf('\n=== Running side %d / %s ===\n', side, sideName);

    labels = string(sourceLabels{side});
    sideRows = fidRows(ismember(fidRows.label, labels), :);
    if height(sideRows) ~= 8
        error('Expected 8 Slicer fiducials for %s, got %d.', sideName, height(sideRows));
    end
    orderedRows = sideRows([], :);
    for labelIdx = 1:numel(labels)
        matchRow = sideRows(sideRows.label == labels(labelIdx), :);
        if height(matchRow) ~= 1
            error('Expected exactly one fiducial row for %s / %s, got %d.', sideName, labels(labelIdx), height(matchRow));
        end
        orderedRows = [orderedRows; matchRow]; %#ok<AGROW>
    end
    sideRows = orderedRows;

    points = [sideRows.anchorNative_ras_x, sideRows.anchorNative_ras_y, sideRows.anchorNative_ras_z];
    [axisPoint, unitvector3, residuals] = fit_axis(points);

    k0Rows = levelRows(strcmp(string(levelRows.lead), sideName) & contains(string(levelRows.level_name), 'K0'), :);
    if height(k0Rows) ~= 1
        error('Expected exactly one K0 row for %s in levels CSV, got %d.', sideName, height(k0Rows));
    end
    head3 = [k0Rows.anchorNative_ras_x(1), k0Rows.anchorNative_ras_y(1), k0Rows.anchorNative_ras_z(1)];
    head_mm = [head3(:); 1];
    unitvector_mm = [unitvector3(:); 0];

    fprintf('K0/head_mm anchorNative RAS: [%.6f %.6f %.6f]\n', head3(1), head3(2), head3(3));
    fprintf('Slicer axis unitvector: [%.9f %.9f %.9f]\n', unitvector3(1), unitvector3(2), unitvector3(3));
    fprintf('Slicer axis fit RMS residual: %.6f mm, max residual: %.6f mm\n', sqrt(mean(residuals.^2)), max(residuals));

    try
        [roll_y, y, solution] = ea_diode_medtronic(side, ct, head_mm, unitvector_mm, tmat_vx2mm, elspec);
        status = "ok";
        errMsg = "";
    catch ME
        roll_y = [];
        y = [];
        solution = struct();
        status = "error";
        errMsg = string(getReport(ME, 'extended', 'hyperlinks', 'off'));
        fprintf('ERROR while running ea_diode_medtronic for %s:\n%s\n', sideName, errMsg);
    end

    res = struct();
    res.side = side;
    res.side_name = sideName;
    res.status = char(status);
    res.error_message = char(errMsg);
    res.head_mm = head_mm;
    res.unitvector_mm = unitvector_mm;
    res.axis_point_mm = axisPoint(:);
    res.axis_fit_rms_residual_mm = sqrt(mean(residuals.^2));
    res.axis_fit_max_residual_mm = max(residuals);
    res.roll_y_rad = roll_y;
    res.roll_y_deg = rad2deg(roll_y);
    res.y_mm = y;
    res.solution = solution;
    results.(sideName) = res;

    if isempty(roll_y)
        rollDegOut = NaN;
    else
        rollDegOut = rad2deg(roll_y);
    end
    if isempty(y)
        y1 = NaN; y2 = NaN; y3 = NaN;
    else
        y1 = y(1); y2 = y(2); y3 = y(3);
    end
    summaryRows(end+1, :) = {side, sideName, char(status), rollDegOut, head3(1), head3(2), head3(3), unitvector3(1), unitvector3(2), unitvector3(3), y1, y2, y3}; %#ok<AGROW>
end

matPath = fullfile(cfg.OutputDir, 'sub-PatientExample_B33005_diode_from_slicer_axis.mat');
save(matPath, 'results', 'cfg', 'elspec');

summary = cell2table(summaryRows, 'VariableNames', { ...
    'side', 'side_name', 'status', 'roll_y_deg', ...
    'head_x', 'head_y', 'head_z', ...
    'axis_x', 'axis_y', 'axis_z', ...
    'y_x', 'y_y', 'y_z'});
csvPath = fullfile(cfg.OutputDir, 'sub-PatientExample_B33005_diode_from_slicer_axis_summary.csv');
writetable(summary, csvPath);

jsonPath = fullfile(cfg.OutputDir, 'sub-PatientExample_B33005_diode_from_slicer_axis_summary.json');
fid = fopen(jsonPath, 'w');
fprintf(fid, '%s', jsonencode(table2struct(summary), 'PrettyPrint', true));
fclose(fid);

fprintf('\nWrote MAT: %s\n', matPath);
fprintf('Wrote CSV: %s\n', csvPath);
fprintf('Wrote JSON: %s\n', jsonPath);
fprintf('Wrote log: %s\n', logPath);

end

function [axisPoint, direction, residuals] = fit_axis(points)
axisPoint = mean(points, 1);
[~, ~, V] = svd(points - axisPoint, 'econ');
direction = V(:, 1)';
if dot(points(end, :) - points(1, :), direction) < 0
    direction = -direction;
end
direction = direction / norm(direction);
rel = points - axisPoint;
proj = (rel * direction') * direction;
perp = rel - proj;
residuals = sqrt(sum(perp.^2, 2));
end
