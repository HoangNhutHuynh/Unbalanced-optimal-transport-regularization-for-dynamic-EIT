%DIAG_E3_P7B  Is patient 7's tidal signal electrode movement rather than gas?
%
%   diag_E3_p7: the tidal difference pattern of patient 1 lies 96 % inside the
%   30 leading singular vectors of the lung-prior Jacobian J (eT 0.96), that
%   of patient 7 only 43-58 % for every electrode mapping and both thorax
%   models - barely above a random vector (sqrt(30/208) = 0.38).
%   Here, per file, the norm fraction of (a) the tidal pattern and (b) the
%   whole centred data matrix explained by
%     J30        30 leading left singular vectors of J
%     Jmov       the 32-column in-plane electrode-movement Jacobian
%     J30+Jmov   both
%   with the random-subspace references sqrt(dim/208). If Jmov (alone or
%   added to J30) explains patient 7 far above its random level while it
%   adds little for patient 1, patient 7's breathing signal is dominated by
%   electrode/boundary movement -> run HK with the movement nuisance
%   (hk_core v7 eta_op, as E2c) or exclude the patient with this reason.
%
%   ~3 min.
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars; clc;
here = fileparts(mfilename('fullpath'));  if isempty(here); here = pwd; end
root = fullfile(here, 'Data', 'child');
pp = {'cg_data_2012_p1/DATA/STUDYNAME/SUBJECT_1/YYYYMMDD/Eit/Viasys/', 'cg_data_2012_p7/DATA/STUDYNAME/SUBJECT_7/YYYYMMDD/Eit/Viasys/'};
FL = {'1001_c1', 1, 'neonate_16el_lungs'; '1001_c4', 1, 'neonate_16el_lungs'; '1001_d4', 1, 'neonate_16el_lungs';
      '1007_b2', 2, 'adult_male_16el_lungs'; '1007_c1', 2, 'adult_male_16el_lungs'; '1007_c4', 2, 'adult_male_16el_lungs'; '1007_d2', 2, 'adult_male_16el_lungs'};
cache = struct();
fprintf('%-8s | %-28s | %-28s | %-28s\n', 'file', 'J30 tidal / all (rand .38)', 'Jmov tidal / all (rand .39)', 'J30+Jmov tidal / all (rand .55)');
for f = 1:size(FL,1)
    D = load_real_eit(fullfile(root, pp{FL{f,2}}, [FL{f,1} '.get']), struct('fps', 13));
    key = strrep(FL{f,3}, '_16el_lungs', '');
    if ~isfield(cache, key)
        fm = mk_library_model(FL{f,3});
        fm.electrode = fm.electrode(electrode_order(fm, 'left'));
        fm.stimulation = D.stim;  fm = mdl_normalize(fm, 1);
        Ne = size(fm.elems, 1);  s0 = ones(Ne,1);  s0([fm.mat_idx{2}(:); fm.mat_idx{3}(:)]) = 0.4;
        J  = calc_jacobian(mk_image(fm, s0));
        Jm = movement_jac(fm, s0, [1 2], 1e-6);
        [U, ~, ~] = svd(J, 'econ');
        Q1 = U(:, 1:30);  [Q2, ~] = qr(Jm, 0);  [Q3, ~] = qr([Q1, Jm], 0);
        cache.(key) = struct('Q1', Q1, 'Q2', Q2, 'Q3', Q3);
    end
    C = cache.(key);
    V = D.vv(:, max(1, end-259):end);  vm = mean(V, 2);
    Y = (V - vm) ./ vm;                                        % normalized difference data
    g = sum(V ./ vm - 1, 1);  [~, o] = sort(g);  n10 = max(3, round(0.1*numel(g)));
    dv = mean(Y(:, o(end-n10+1:end)), 2) - mean(Y(:, o(1:n10)), 2);
    fr = @(Q, x) norm(Q' * x, 'fro') / norm(x, 'fro');
    fprintf('%-8s | %12.3f / %-13.3f | %12.3f / %-13.3f | %12.3f / %-13.3f\n', FL{f,1}, ...
            fr(C.Q1, dv), fr(C.Q1, Y), fr(C.Q2, dv), fr(C.Q2, Y), fr(C.Q3, dv), fr(C.Q3, Y));
end
fprintf('(random references: sqrt(30/208) = %.2f, sqrt(32/208) = %.2f, sqrt(62/208) = %.2f)\n', sqrt(30/208), sqrt(32/208), sqrt(62/208));

% ======================================================================
function Jm = movement_jac(fm, s0, dims, delta)
% finite-difference electrode-movement Jacobian, normalized like J (dV/V)
img  = mk_image(fm, s0);
d0   = fwd_solve(img);  v0 = d0.meas;
node0 = fm.nodes;  ne = numel(fm.electrode);
Jm = zeros(numel(v0), ne * numel(dims));  col = 0;
for d = dims
    for i = 1:ne
        idx = fm.electrode(i).nodes;
        img.fwd_model.nodes = node0;
        img.fwd_model.nodes(idx, d) = node0(idx, d) + delta;
        di = fwd_solve(img);
        col = col + 1;
        Jm(:, col) = (di.meas - v0) / delta ./ v0;
    end
end
end


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

