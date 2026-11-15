%DIAG_E3_P7  Is patient 7's electrode mapping (start electrode, direction) or
%   measurement layout different from what the model assumes?
%
%   diag_E3_quality: patient 7's data are of comparable quality to patient 1
%   (reciprocity error 0.023-0.035 vs 0.019-0.028; ventilation/noise 5-9 vs
%   6-16), yet the L-curve corner sits at the lower edge (hp ~1e-4) for every
%   model, while patient 1 has hp ~1. The fixed-lambda misfit did not
%   discriminate between models (0.82-0.91 for P7 whatever the model).
%   Here the electrode mapping is varied by PERMUTING THE DATA ROWS (no new
%   Jacobian): a cyclic shift of the electrode numbering by k = 0..15 and a
%   reversal of direction (32 mappings), for each library thorax. Metrics:
%     cH  correlation of the mean frame with the homogeneous forward solution
%         (U-shaped adjacent pattern; tests the measurement layout)
%     eT  fraction of the tidal difference pattern (top-10% minus bottom-10%
%         global frames, normalized) inside the 30 leading singular vectors of
%         the lung-prior Jacobian (tests where ventilation sits w.r.t. the
%         electrodes)
%   Patient 1 (1001_c1, documented e1 = sternum, numbering to the left) is
%   the control: its best mapping should be k = 0, 'left'.
%   Note: the 'right' rows reflect the data with the same rule as
%   electrode_order (keep e1, reverse the rest).
%
%   ~2 min.
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars; clc;
here = fileparts(mfilename('fullpath'));  if isempty(here); here = pwd; end
root = fullfile(here, 'Data', 'child');
FL = {'1001_c1', fullfile(root, 'cg_data_2012_p1/DATA/STUDYNAME/SUBJECT_1/YYYYMMDD/Eit/Viasys/1001_c1.get');
      '1007_b2', fullfile(root, 'cg_data_2012_p7/DATA/STUDYNAME/SUBJECT_7/YYYYMMDD/Eit/Viasys/1007_b2.get');
      '1007_c4', fullfile(root, 'cg_data_2012_p7/DATA/STUDYNAME/SUBJECT_7/YYYYMMDD/Eit/Viasys/1007_c4.get')};
models = {'neonate_16el_lungs', 'adult_male_16el_lungs'};
K = 30;

for mi = 1:numel(models)
    fm = mk_library_model(models{mi});
    fm.electrode = fm.electrode(electrode_order(fm, 'left'));
    D0 = load_real_eit(FL{1,2}, struct('fps', 13));
    fm.stimulation = D0.stim;  fm = mdl_normalize(fm, 1);
    Ne = size(fm.elems, 1);  s0 = ones(Ne,1);  s0([fm.mat_idx{2}(:); fm.mat_idx{3}(:)]) = 0.4;
    img = mk_image(fm, s0);
    J  = calc_jacobian(img);
    vs = fwd_solve(img);  vs = vs.meas;
    [U, ~, ~] = svd(J, 'econ');  Uk = U(:, 1:K);
    [pairs, nel] = meas_pairs(D0.stim);
    fprintf('\n==== model %s (rows: data file; best mapping first) ====\n', models{mi});
    for f = 1:size(FL,1)
        D = load_real_eit(FL{f,2}, struct('fps', 13));
        V = D.vv(:, max(1, end-259):end);  vm = mean(V, 2);
        g = sum(V ./ vm - 1, 1);  [~, o] = sort(g);  n10 = max(3, round(0.1*numel(g)));
        dv = (mean(V(:, o(end-n10+1:end)), 2) - mean(V(:, o(1:n10)), 2)) ./ vm;
        R = zeros(2*nel, 4);  r = 0;
        for dirn = [1 -1]
            for k = 0:nel-1
                p = permute_rows(pairs, nel, k, dirn);
                if any(p == 0); continue; end
                r = r + 1;
                cH = corr_(abs(vm(p)), abs(vs));
                eT = norm(Uk' * dv(p)) / norm(dv);
                R(r,:) = [k, dirn, cH, eT];
            end
        end
        R = R(1:r, :);  [~, ob] = sort(R(:,4), 'descend');  R = R(ob, :);
        i0 = find(R(:,1) == 0 & R(:,2) == 1);
        fprintf('%-8s | documented (k 0, left): cH %.3f eT %.3f (rank %d of %d) | best: ', FL{f,1}, R(i0,3), R(i0,4), i0, r);
        for b = 1:3
            fprintf('k %2d %-5s cH %.3f eT %.3f;  ', R(b,1), ternary(R(b,2) == 1, 'left', 'right'), R(b,3), R(b,4));
        end
        fprintf('\n');
    end
end
fprintf(['\nReading: if patient 7''s eT is much lower than patient 1''s for EVERY mapping, its tidal signal is not\n' ...
         'explained by the 16-electrode plane model at all (wrong plane / electrode belt, ECMO interference);\n' ...
         'if one mapping clearly wins, patient 7''s electrodes were numbered differently from the documentation.\n']);

% ======================================================================
function [P, nel] = meas_pairs(stim)
% for each measurement row: [drive+, drive-, meas+, meas-] electrode indices
nel = numel(stim(1).stim_pattern);  P = [];
for k = 1:numel(stim)
    d = full(stim(k).stim_pattern(:));  M = full(stim(k).meas_pattern);
    dp = find(d > 0, 1);  dm = find(d < 0, 1);
    for j = 1:size(M, 1)
        P = [P; dp, dm, find(M(j,:) > 0, 1), find(M(j,:) < 0, 1)]; %#ok<AGROW>
    end
end
end

function p = permute_rows(P, nel, k, dirn)
% row index in the data of the measurement that the model's row i sees when
% the data numbering is shifted by k (and reversed for dirn = -1)
map = @(e) mod(dirn * (e - 1) + k, nel) + 1;
key = @(Q) sort(Q(:,1:2), 2) * [nel^3; nel^2] + sort(Q(:,3:4), 2) * [nel; 1];
kd = key(P);
Q  = [map(P(:,1)), map(P(:,2)), map(P(:,3)), map(P(:,4))];
kq = key(Q);
[tf, p] = ismember(kq, kd);
p(~tf) = 0;
end

function c = corr_(a, b)
a = a - mean(a);  b = b - mean(b);  c = (a' * b) / max(norm(a) * norm(b), eps);
end

function s = ternary(c, a, b); if c; s = a; else; s = b; end; end

function [ant, rgt, c_t] = axes_dir(fm)
ce  = interp_mesh(fm, 0);  vol = get_elem_volume(fm);
wc  = @(ix) sum(ce(ix,1:2) .* vol(ix), 1) / sum(vol(ix));
c_t = wc(1:size(fm.elems,1));  c_l = wc(fm.mat_idx{2});  c_r = wc(fm.mat_idx{3});
rgt = (c_r - c_l) / norm(c_r - c_l);
ant = c_t - (c_l + c_r)/2;  ant = ant - dot(ant, rgt)*rgt;  ant = ant / norm(ant);
end

function order = electrode_order(fm, towards)
[ant, rgt, c_t] = axes_dir(fm);
ep  = zeros(numel(fm.electrode), 2);
for k = 1:numel(fm.electrode); ep(k,:) = mean(fm.nodes(fm.electrode(k).nodes, 1:2), 1) - c_t; end
a   = atan2(ep*rgt', ep*ant');
[~, k0] = min(abs(a));  n = size(ep, 1);
order = circshift(1:n, -(k0-1));
side = dot(ep(order(5),:), rgt);
if (strcmp(towards, 'left') && side > 0) || (strcmp(towards, 'right') && side <= 0)
    order = order([1, n:-1:2]);
end
end

