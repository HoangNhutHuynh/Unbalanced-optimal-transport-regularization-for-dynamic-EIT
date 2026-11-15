%RUN_E1_SENS  E1 sensitivity analysis (closes E1): spatial prior beta.
%
%   E1 v2 showed that in the 2-plane model the HK image suppresses the
%   cranio-caudal anti-phase swing (HK/GN swing ratio ~0.28): the spatial
%   prior beta*||D rho||^2 (beta = 1e-2, tuned on 1-plane simulations)
%   penalizes exactly the vertical gradient across the planes, so the shift
%   collapses to a global component that the ventilation mode absorbs.
%
%   Here: beta in {1e-4, 1e-3, 1e-2}, both models, records 001 (tidal) and
%   003 (paradoxical), window 1 (20 s, x2 downsampled). Everything else is
%   protocol P1 as in run_E1_hk v2, except
%     rho0 = kappa * GLOBAL tidal TIV per unit volume, from the UNCLIPPED TIV
%            map (the volume-weighted sum is the most stable quantity of an
%            EIT image; v2 used the clipped map -> rho0 varied 28x between
%            models through its hp dependence).
%   GT-free consistency criterion: the HK image must keep the regional swing
%   that the linear GN image shows (swing ratio HK/GN). A conclusion counts
%   as ROBUST only if it holds for every beta with swing ratio >= 0.7.
%
%   ~35 min. Needs E1_prep.mat and EIDORS.
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars; clc;
here = fileparts(mfilename('fullpath'));  if isempty(here); here = pwd; end
S = load(fullfile(here, 'E1_prep.mat'));
recs = S.recs_save;  fps = S.fps;

betas = [1e-4 1e-3 1e-2];
ids   = {'001', '003'};
win_s = 20;  ds = 2;  dt = ds / fps;
P1 = struct('alpha', 1e-3, 'dm', 4, 'kappa', 3);
sig_lung = 0.4;  N_coarse = 800;  tau0 = 100;  it_hk = 4000;
models = {S.fmdl1, S.fmdl2};

R = struct('plane', {}, 'id', {}, 'beta', {}, 'iface', {}, 'ratio', {}, 'Tshare', {}, ...
           'Vshare', {}, 'Rg_hk', {}, 'Rg_gn', {}, 'PVflux', {});
for pl = 1:2
    fm  = models{pl};
    Ne  = size(fm.elems, 1);
    lung_idx = [fm.mat_idx{2}(:); fm.mat_idx{3}(:)];
    s0 = ones(Ne, 1);  s0(lung_idx) = sig_lung;
    J  = calc_jacobian(mk_image(fm, s0));
    grp = zeros(Ne,1);  grp(fm.mat_idx{2}) = 1;  grp(fm.mat_idx{3}) = 2;
    Gc = calc_dual_graph(fm, lung_idx, struct('target_N', N_coarse, 'groups', grp));
    Ac = J * Gc.EL;
    yl  = Gc.cent(:,2);
    ifc = struct('name', {}, 'r1', {}, 'L', {});
    ifc(end+1) = struct('name', 'anterior|posterior', 'r1', yl > wmedian(yl, Gc.vol), 'L', NaN);
    if pl == 2
        ifc(end+1) = struct('name', 'cranial|caudal', ...
                            'r1', (Gc.cent(:,3) - 0.5) * sign(S.sup_z) > 0, 'L', NaN);
    end
    for f = 1:numel(ifc)
        r1 = ifc(f).r1;
        ifc(f).L = norm(sum(Gc.cent(r1,:).*Gc.vol(r1),1)/sum(Gc.vol(r1)) - ...
                        sum(Gc.cent(~r1,:).*Gc.vol(~r1),1)/sum(Gc.vol(~r1)));
    end
    main = numel(ifc);

    % tidal profile (clipped) and rho0 (from the UNCLIPPED global TIV)
    rt  = recs([recs.plane] == pl & strcmp({recs.id}, '001'));
    Yt  = calc_difference_data(rt.vh_low, rt.V_lp(:, 1:ds:end), fm);
    hpt = lcurve_hp(Ac, Yt);
    rgt = -gn_frames(Ac, Yt, hpt);
    gt  = Gc.vol' * rgt;  [~, o] = sort(gt);  n10 = max(3, round(0.1*numel(gt)));
    dtiv = mean(rgt(:, o(end-n10+1:end)), 2) - mean(rgt(:, o(1:n10)), 2);
    w_tidal = max(dtiv, 0);
    rho0 = P1.kappa * max(sum(Gc.vol .* dtiv) / sum(Gc.vol), 0);
    fprintf('\n######## %d-plane: tidal hp %.2g | rho0 (unclipped global TIV) %.4g\n', pl, hpt, rho0);

    for ii = 1:numel(ids)
        r = recs([recs.plane] == pl & strcmp({recs.id}, ids{ii}));
        if strcmp(r.kind, 'paradox'); vh = r.vh_mean; else; vh = r.vh_low; end
        fr  = 1:ds:round(win_s * fps);
        Yd  = calc_difference_data(vh, r.V_lp(:, fr), fm);
        hp  = lcurve_hp(Ac, Yd);
        rgn = -gn_frames(Ac, Yd, hp);
        for beta = betas
            o = struct('alpha', P1.alpha, 'beta', beta, 'delta', P1.dm*ifc(main).L/pi, ...
                       'gamma', 1e3, 'dt', dt, 'max_iter', it_hk, 'tol', 1e-4, 'verbose', 0, ...
                       'use_eta', false, 'normalize', true, 'tau0', tau0, ...
                       'rho0', rho0, 'vent_w', w_tidal);
            [rho, m, ~, info] = hk_core(-J, speye(size(J,1)), Yd, Gc, o);
            gsrc = dt * (Gc.vol' * info.zeta);
            gV   = Gc.vol' * rho;
            Vv   = [0, cumsum(dt * info.vent_c)];
            Vsh  = p2p(Vv) / max(p2p(gV), eps);
            for f = 1:numel(ifc)
                r1  = ifc(f).r1;
                QG  = hk_pendelluft(Gc, m, r1, dt);
                swh = p2p((Gc.vol(r1)' * rho(r1,:) - Gc.vol(~r1)' * rho(~r1,:)) / 2);
                swg = p2p((Gc.vol(r1)' * rgn(r1,:) - Gc.vol(~r1)' * rgn(~r1,:)) / 2);
                Tsh = sum(abs(dt*QG)) / (sum(abs(dt*QG)) + sum(abs(gsrc)) + eps);
                R(end+1) = struct('plane', pl, 'id', ids{ii}, 'beta', beta, 'iface', ifc(f).name, ...
                                  'ratio', swh / max(swg, eps), 'Tshare', Tsh, 'Vshare', Vsh, ...
                                  'Rg_hk', p2p(gV) / max(swh, eps), 'Rg_gn', p2p(Gc.vol' * rgn) / max(swg, eps), ...
                                  'PVflux', dt * sum(max(QG, 0))); %#ok<SAGROW>
                fprintf(['%dpl %s beta %.0e | %-18s | swing HK/GN %.2f | T-share %.2f | vent-share %.2f | ' ...
                         'Rg HK %.2f GN %.2f | flux-PV %.3e | res %.0e\n'], pl, ids{ii}, beta, ...
                         ifc(f).name, R(end).ratio, Tsh, Vsh, R(end).Rg_hk, R(end).Rg_gn, ...
                         R(end).PVflux, max(info.res));
            end
        end
    end
end
save(fullfile(here, 'E1_sens_results.mat'), 'R', 'betas', 'P1');

% ---------------- robustness summary ----------------
fprintf('\n============ E1 ROBUSTNESS (only beta with swing ratio >= 0.7 for BOTH records) ============\n');
fprintf('%-5s %-18s %-12s %10s %10s %10s %10s %s\n', 'model', 'interface', 'valid betas', ...
        'Tsh 001', 'Tsh 003', 'Vsh 001', 'Vsh 003', 'T-share up & vent-share down in 003?');
tag = unique(cellfun(@(a,b) sprintf('%d|%s', a, b), {R.plane}, {R.iface}, 'UniformOutput', false), 'stable');
for k = 1:numel(tag)
    sel = strcmp(cellfun(@(a,b) sprintf('%d|%s', a, b), {R.plane}, {R.iface}, 'UniformOutput', false), tag{k});
    Rk = R(sel);  vb = [];  T1 = [];  T3 = [];  V1 = [];  V3 = [];
    for beta = betas
        a = Rk([Rk.beta] == beta & strcmp({Rk.id}, '001'));
        b = Rk([Rk.beta] == beta & strcmp({Rk.id}, '003'));
        if isempty(a) || isempty(b); continue; end
        if a.ratio >= 0.7 && b.ratio >= 0.7
            vb(end+1) = beta;  T1(end+1) = a.Tshare;  T3(end+1) = b.Tshare; %#ok<AGROW>
            V1(end+1) = a.Vshare;  V3(end+1) = b.Vshare; %#ok<AGROW>
        end
    end
    if isempty(vb)
        fprintf('%-5s %-18s %-12s  (no beta keeps the GN swing -> no conclusion)\n', ...
                sprintf('%dpl', Rk(1).plane), Rk(1).iface, '-');
    else
        ok = all(T3 > T1) && all(V3 < V1);
        fprintf('%-5s %-18s %-12s %10s %10s %10s %10s %s\n', sprintf('%dpl', Rk(1).plane), Rk(1).iface, ...
                mat2str(vb), mat2str(T1, 2), mat2str(T3, 2), mat2str(V1, 2), mat2str(V3, 2), ...
                ternary(ok, 'YES (robust)', 'no'));
    end
end

% ======================================================================
function v = p2p(x); v = max(x) - min(x); end

function s = ternary(c, a, b); if c; s = a; else; s = b; end; end

function [hp, edge] = lcurve_hp(A, Y)
p = 1 ./ sqrt(sum(A.^2, 1))';
S = A * (p .* A');  S = (S + S')/2;
[U, D] = eig(S);  s = max(diag(D), 0);
Z2 = sum((U' * Y).^2, 2);
lams = logspace(-8, 2, 101) * mean(s);
rr = zeros(size(lams));  xx = zeros(size(lams));
for i = 1:numel(lams)
    rr(i) = sum((lams(i) ./ (s + lams(i))).^2 .* Z2);
    xx(i) = sum(s .* Z2 ./ (s + lams(i)).^2);
end
x = 0.5*log(rr);  y = 0.5*log(xx);  t = log(lams);
x1 = gradient(x, t);  y1 = gradient(y, t);  x2 = gradient(x1, t);  y2 = gradient(y1, t);
kap = (x1.*y2 - x2.*y1) ./ max((x1.^2 + y1.^2).^1.5, eps);
kap([1:3, end-2:end]) = -Inf;
[~, k] = max(kap);  hp = sqrt(lams(k));
edge = k <= 4 || k >= numel(lams) - 3;
end

function m = wmedian(x, w)
[xs, o] = sort(x);  cw = cumsum(w(o)) / sum(w);
m = xs(find(cw >= 0.5, 1));
end

function X = gn_frames(A, Y, hp)
p   = 1 ./ sqrt(sum(A.^2, 1))';
PAt = p .* A';
X   = PAt * ((A*PAt + hp^2*eye(size(A,1))) \ Y);
end
