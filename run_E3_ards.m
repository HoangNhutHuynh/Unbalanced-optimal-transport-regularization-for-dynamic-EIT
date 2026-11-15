%RUN_E3_ARDS  (v2) E3: quadrature HK pendelluft/asynchrony index across a PEEP
%   recruitment manoeuvre and titration in paediatric ARDS.
%
%   Data: EIDORS contrib "cg-2012-ards-recruitment" (Children's Hospital
%   Boston 2009, NCT00830284). Goe-MF II (Viasys/CareFusion), 16 electrodes,
%   adjacent drive/measure, 6th intercostal space; electrode 1 on the sternum,
%   numbering towards the patient's LEFT. Frame rate 13 Hz (Viasys default, as
%   in Gomez-Laberge's EITReadData; no .prl). Mechanically ventilated.
%     Patient 1: F, 5.9 y, 20 kg, primary ARDS (parainfluenza pneumonia)
%     Patient 7: F, 13.9 y, 76 kg, secondary ARDS (sepsis), on ECMO
%   Cite: Gomez-Laberge C, Arnold JH, Wolf GK, IEEE TMI 31:834-842, 2012;
%         Wolf GK et al., Pediatr Crit Care Med 13:509-515, 2012.
%   ASSUMPTIONS (not stated on the data page): supine position, so the
%   POSTERIOR region is gravity-dependent; no library model of a child thorax
%   exists in EIDORS, so patient 1 uses 'neonate_16el_lungs' (infant thorax
%   shape) and patient 7 'adult_male_16el_lungs' (adult-sized adolescent).
%
%   Passive mechanical ventilation produces no effort-driven pendelluft, so
%   the quadrature HK volume measures lagged regional redistribution, i.e.
%   regional asynchrony (time-constant heterogeneity, tidal recruitment) in
%   its transfer-equivalent form (E2g/E2h). Hypotheses:
%     H1  the index is above its in-phase null floor at every PEEP
%     H2  it changes systematically with PEEP within a patient (Spearman)
%     H3  at low PEEP the dependent (posterior) region lags: lag index
%         L = (P>A early + A>P late - A>P early - P>A late) / sum > 0
%         (E2g v2 timing rule: the LATER region gives outward flux early and
%         inward flux late; P>A = posterior -> anterior)
%
%   Per file: the last WIN_S seconds of the artefact-free range (steady state
%   at the end of each step; ranges from Script_For_Tutorial for patient 1),
%   cardiac low-pass, breaths from GN gas-volume minima, HK P1 (no nuisance),
%   raw and quadrature PV per breath (% of GN TV), timing thirds, asymmetry,
%   lag index; an in-phase null (zero transfer, real residual phase-
%   randomized) for the files in RUN_NULL.
%   RESUMABLE: one result file per recording in E3_results\; existing results
%   are skipped, so the run can be stopped and restarted.
%   MASK: 'lungs' | 'dilate2' | 'thorax'. Set to 'thorax' after E2i: on the
%   neonate the GN and HK tidal images put 64-72 % of the TIV OUTSIDE the
%   library lungs, the whole-thorax HK fits as well as GN (misfit 0.130 vs
%   0.318 with lungs only) and the positivity bound is almost never active;
%   2-layer dilation already covers 90 % of the thorax on these coarse meshes.
%
%   v2 (after v1, Steve 2026-11-09): (1) the cardiac peak is accepted only if
%   it is prominent (peak / band median >= 4) and above 2.5 RR; otherwise no
%   cardiac filter is assumed and the low-pass keeps three respiratory
%   harmonics (fc = 3 RR) - v1 put fc BELOW the 2nd harmonic in patient 1;
%   (2) 40-s windows (more breaths; v1 had 3-7); (3) in-phase nulls for 5 of
%   the 9 patient-1 steps (v1: null ~1 %TV, real only ~2x above); (4)
%   SUBJ_RUN selects the patients (patient 7 failed in v1: data not
%   representable by the model - see diag_E3_quality); results *_v2.mat.
%   ~5 h for patient 1 (9 recordings + 5 nulls, ~20 min each).
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars; clc;
here = fileparts(mfilename('fullpath'));  if isempty(here); here = pwd; end
root   = fullfile(here, 'Data', 'child');
outdir = fullfile(here, 'E3_results');  if ~exist(outdir, 'dir'); mkdir(outdir); end

MASK = 'thorax';     % E2i: library lung mask holds only ~1/3 of the reconstructed TIV;
                     % the whole-thorax domain fits like GN (misfit 0.130 vs 0.318) with bound <= 1 %
WIN_S = 40;
RUN_NULL = {'1001_b', '1001_c2', '1001_c4', '1001_d1', '1001_d4', '1007_b2', '1007_c4'};
MAPS_ONLY = {};  % e.g. {'1001_b'}:   % non-empty: only these files, real HK only, maps for Fig16 -> maps\E3_<id>.mat ({} = full E3)
SUBJ_RUN = 1;          % patient 7 excluded until diag_E3_quality explains the v1 failure
P1 = struct('alpha', 1e-3, 'beta', 1e-2, 'dm', 4, 'kappa', 3);
sig_lung = 0.4;  N_lung = 800;  N_max = 1000;  tau0 = 100;  it_hk = 6000;  fps0 = 13;   % N_max 1000: ~13 min/run over the thorax

p1 = 'cg_data_2012_p1/DATA/STUDYNAME/SUBJECT_1/YYYYMMDD/Eit/Viasys/';
p7 = 'cg_data_2012_p7/DATA/STUDYNAME/SUBJECT_7/YYYYMMDD/Eit/Viasys/';
%        id        subject  path  maneuver     PEEP dP   range (frames, [] = all)
F = {'1001_b',    1, p1, 'baseline',   14,  5,  [];
     '1001_c1',   1, p1, 'increment',  15, 15,  [];
     '1001_c2',   1, p1, 'increment',  20, 15,  [];
     '1001_c3',   1, p1, 'increment',  25, 15,  [];
     '1001_c4',   1, p1, 'increment',  30, 15,  [];
     '1001_d1',   1, p1, 'decrement',  20,  6,  [1 765];
     '1001_d2',   1, p1, 'decrement',  18,  6,  [];
     '1001_d3',   1, p1, 'decrement',  16,  5,  [368 780];
     '1001_d4',   1, p1, 'decrement',  14,  4,  [411 780];
     '1007_b2',   7, p7, 'baseline',   22, 16,  [];
     '1007_c1',   7, p7, 'increment',  15, 15,  [];
     '1007_c2',   7, p7, 'increment',  20, 15,  [];
     '1007_c3',   7, p7, 'increment',  25, 15,  [];
     '1007_c4',   7, p7, 'increment',  30, 15,  [];
     '1007_d1',   7, p7, 'decrement',  20, 17,  [];
     '1007_d2',   7, p7, 'decrement',  18, 15,  []};
model_of = containers.Map({1, 7}, {'neonate_16el_lungs', 'adult_male_16el_lungs'});

MD = struct();                                     % per-subject model cache
for f = 1:size(F, 1)
    id = F{f,1};  subj = F{f,2};
    if ~ismember(subj, SUBJ_RUN); continue; end
    rfile = fullfile(outdir, sprintf('%s_%s_v2.mat', id, MASK));
    if ~isempty(MAPS_ONLY)
        if ~any(strcmp(MAPS_ONLY, id)); continue; end
        if ~exist(fullfile(here, 'maps'), 'dir'); mkdir(fullfile(here, 'maps')); end
        rfile = fullfile(here, 'maps', ['E3_' id '.mat']);
    end
    if exist(rfile, 'file'); fprintf('%s: done (skipped)\n', id); continue; end
    t_file = tic;
    % ---- data, window ----
    D = load_real_eit(fullfile(root, F{f,3}, [id '.get']), struct('fps', fps0));
    fps = D.fps;  dt = 1/fps;
    rg = F{f,7};  if isempty(rg); rg = [1 size(D.vv, 2)]; end
    nW = min(round(WIN_S * fps), rg(2) - rg(1) + 1);
    fr = rg(2) - nW + 1 : rg(2);
    V  = D.vv(:, fr);  T = size(V, 2);
    fprintf('\n%s (P%d, %s, PEEP %d, dP %d): frames %d-%d of %d | bad %d, flagged ch %d, jumps %d\n', id, subj, ...
            F{f,4}, F{f,5}, F{f,6}, fr(1), fr(end), size(D.vv,2), numel(D.qc.bad_frames), ...
            numel(D.qc.chan_flag), numel(D.qc.jump_frames));
    g = sum(V ./ mean(V, 2) - 1, 1);
    [S_, frq] = spectrum(g, fps);
    RR = peak_in(S_, frq, [0.15 1.0]);
    hb = frq >= max(1.6*RR, 0.75) & frq <= min(3.5, 0.45*fps);
    [pk, kh] = max(S_ .* hb);  HR = frq(kh);  prom = pk / median(S_(hb));
    if prom >= 4 && 0.8*HR > 2.5*RR
        fc = min(0.8*HR, max(2.5*RR, 0.5*(2.5*RR + 0.8*HR)));
    else
        HR = NaN;  fc = min(3*RR, 0.4*fps);      % no reliable cardiac peak: keep 3 respiratory harmonics
    end
    V_lp = fft_lowpass(V, fps, fc, 0.2*fc);
    % ---- model (cached per subject) ----
    key = sprintf('s%d', subj);
    if ~isfield(MD, key)
        fm = mk_library_model(model_of(subj));
        fm.electrode = fm.electrode(electrode_order(fm, 'left'));
        fm.stimulation = D.stim;  fm = mdl_normalize(fm, 1);
        Ne = size(fm.elems, 1);
        isL = false(Ne,1);  isL(fm.mat_idx{2}) = true;  isR = false(Ne,1);  isR(fm.mat_idx{3}) = true;
        isLung = isL | isR;
        s0 = ones(Ne,1);  s0(isLung) = sig_lung;
        J  = calc_jacobian(mk_image(fm, s0));
        vol = get_elem_volume(fm);
        switch MASK
            case 'lungs';   msk = isLung;
            case 'dilate2'
                E2N = sparse(repmat((1:Ne)', 1, size(fm.elems,2)), fm.elems, 1, Ne, size(fm.nodes,1));
                msk = isLung;  for i = 1:2; msk = (E2N * (E2N' * double(msk))) > 0; end
            case 'thorax';  msk = true(Ne,1);
        end
        grp = 3 * ones(Ne,1);  grp(isL) = 1;  grp(isR) = 2;
        Nc  = min(N_max, round(N_lung * sum(vol(msk)) / sum(vol(isLung))));
        Gc  = calc_dual_graph(fm, find(msk), struct('target_N', Nc, 'groups', grp));
        [ant, ~, c_t] = axes_dir(fm);
        ant_c = (Gc.cent(:,1:2) - c_t) * ant' > 0;
        L_ap  = norm(sum(Gc.cent(ant_c,:).*Gc.vol(ant_c),1)/sum(Gc.vol(ant_c)) - ...
                     sum(Gc.cent(~ant_c,:).*Gc.vol(~ant_c),1)/sum(Gc.vol(~ant_c)));
        MD.(key) = struct('fm', fm, 'J', J, 'Gc', Gc, 'Ac', J * Gc.EL, 'ant_c', ant_c, 'L_ap', L_ap);
        fprintf('  model %s, mask %s: %d elements, %d clusters\n', model_of(subj), MASK, Ne, Gc.N);
    end
    M = MD.(key);  Gc = M.Gc;  Ac = M.Ac;  J = M.J;  ant_c = M.ant_c;
    % ---- breaths (GN gas volume, two passes) ----
    minsep = round(0.6 * fps / RR);
    vh = mean(V_lp, 2);
    for pass = 1:2
        Y   = calc_difference_data(vh, V_lp, M.fm);
        hp  = lcurve_hp(Ac, Y);
        rgn = -gn_frames(Ac, Y, hp);
        gV  = Gc.vol' * rgn;
        kmin = find(islocalmin(gV, 'MinSeparation', minsep, 'MinProminence', 0.2*p2p(gV)));
        vh  = mean(V_lp(:, kmin), 2);
    end
    nb = numel(kmin) - 1;
    if nb < 3
        fprintf('  only %d complete breaths - skipped\n', nb);  continue;
    end
    kei = zeros(1, nb);
    for b = 1:nb; kk = kmin(b):kmin(b+1);  [~, ip] = max(gV(kk));  kei(b) = kk(ip); end
    kmin = kmin(:)';
    TVg  = arrayfun(@(b) gV(kei(b)) - gV(kmin(b)), (1:nb)');
    tivm = mean(rgn(:, kei) - rgn(:, kmin(1:nb)), 2);
    w    = max(tivm, 0);
    rho0 = P1.kappa * max(sum(Gc.vol .* tivm) / sum(Gc.vol), 0);
    fprintf('  RR %.0f /min, HR %.0f /min (cardiac prominence %.1f), fc %.2f Hz | %d breaths, insp %.2f s | GN hp %.3g\n', ...
            60*RR, 60*HR, prom, fc, nb, median(kei - kmin(1:nb))*dt, hp);
    o = struct('alpha', P1.alpha, 'beta', P1.beta, 'delta', P1.dm*M.L_ap/pi, 'gamma', 1e3, ...
               'dt', dt, 'max_iter', it_hk, 'tol', 1e-4, 'verbose', 0, 'use_eta', false, ...
               'normalize', true, 'tau0', tau0, 'rho0', rho0, 'vent_w', w);
    conds = {'real'};
    if any(strcmp(RUN_NULL, id)) && isempty(MAPS_ONLY); conds{end+1} = 'null-w'; end %#ok<AGROW>
    res = struct('id', id, 'subject', subj, 'maneuver', F{f,4}, 'PEEP', F{f,5}, 'dP', F{f,6}, ...
                 'frames', fr, 'fps', fps, 'RR', RR, 'HR', HR, 'fc', fc, 'hp', hp, 'kmin', kmin, ...
                 'kei', kei, 'TVg', TVg, 'mask', MASK, 'model', model_of(subj));
    for c = 1:numel(conds)
        if c == 1
            Yk = Y;
        else
            Yk = -Ac * (w * (gV / (Gc.vol' * w))) + phase_rand(Y + Ac*rgn, 1);
        end
        t0 = tic;
        [rho, m, ~, info] = hk_core(-J, speye(size(J,1)), Yk, Gc, o);
        QG = hk_pendelluft(Gc, m, ant_c, dt);
        Fl = (diff(Gc.vol' * rho) / dt)';
        a  = (Fl' * QG) / (Fl' * Fl);  r = QG - a * Fl;  cc = corrcoef(QG, Fl);
        Bq = 100 * insp_pv(r,  kmin, kei, dt) ./ TVg;
        Br = 100 * insp_pv(QG, kmin, kei, dt) ./ TVg;
        TT = timing_thirds(r, kmin, kei, dt, TVg);
        Lg = (TT(1) + TT(6) - TT(4) - TT(3)) / max(sum(TT([1 3 4 6])), eps);
        hV = Gc.vol' * rho;
        TVh = arrayfun(@(b) hV(kei(b)) - hV(kmin(b)), (1:nb)');
        act = (rho + rho0) <= 1e-3 * rho0;
        R = struct('Bq', Bq, 'Braw', Br, 'TT', TT, 'lag', Lg, 'a', a, 'cc', cc(1,2), 'QG', QG, 'Fl', Fl, ...
                   'mis', norm(-Ac*rho - Yk, 'fro')/norm(Yk, 'fro'), 'misGN', norm(-Ac*rgn - Y, 'fro')/norm(Y, 'fro'), ...
                   'bound', mean(act(:)), 'tvr', median(TVh ./ TVg), 'res', max(info.res), 'iters', info.iters);
        res.(strrep(conds{c}, '-', '_')) = R;
        if ~isempty(MAPS_ONLY)
            MP = pend_maps(Gc, rgn, rho, m, kmin, kei, ant_c, dt);
            MP.label = sprintf('Paediatric ARDS P1, %s (PEEP %d)', F{f,4}, F{f,5});
            fm = M.fm;  save(rfile, 'MP', 'fm', 'Gc');  %#ok<NASGU>
            fprintf('  maps saved %s (HK quadrature %.2f %%TV)\n', rfile, median(MP.PVq));
        end
        mq = median(Bq, 1, 'omitnan');
        fprintf(['  %-6s | quadrature P>A %.2f A>P %.2f %%TV (raw %.2f %.2f) | asym %.2f | lag index %+.2f | ' ...
                 'thirds P>A %.2f %.2f %.2f A>P %.2f %.2f %.2f | a %+.4f corr %+.2f | misfit %.3f (GN %.3f) bound %.3f TV %.2f | %d it %.0f s\n'], ...
                 conds{c}, mq, median(Br, 1, 'omitnan'), mq(2)/max(mq(1), eps), Lg, TT, a, cc(1,2), ...
                 R.mis, R.misGN, R.bound, R.tvr, info.iters, toc(t0));
    end
    if ~isempty(MAPS_ONLY); continue; end
    save(rfile, 'res');
    fprintf('  saved %s (%.0f min)\n', rfile, toc(t_file)/60);
end

% ---------------- summary over all saved recordings ----------------
fprintf('\n============ E3 summary (mask %s; P>A = posterior(dependent) -> anterior; %%TV per inspiration) ============\n', MASK);
fprintf('%-8s %-9s %4s %3s %3s %5s | %6s %6s %6s %6s | %6s %6s %6s | %5s %6s | %s\n', 'file', 'maneuver', 'PEEP', 'dP', 'nb', 'RR', ...
        'P>A', 'IQR', 'A>P', 'IQR', 'nullPA', 'nullAP', 'nulLag', 'asym', 'lag', 'a / corr');
S = struct('subj', {}, 'PEEP', {}, 'tot', {}, 'lag', {});
for f = 1:size(F, 1)
    rfile = fullfile(outdir, sprintf('%s_%s_v2.mat', F{f,1}, MASK));
    if ~exist(rfile, 'file'); continue; end
    L = load(rfile);  r = L.res;  q = r.real.Bq;
    mq = median(q, 1, 'omitnan');  iq = [prc(q(:,1), 75) - prc(q(:,1), 25), prc(q(:,2), 75) - prc(q(:,2), 25)];
    if isfield(r, 'null_w'); nn = [median(r.null_w.Bq, 1, 'omitnan'), r.null_w.lag]; else; nn = [NaN NaN NaN]; end
    fprintf('%-8s %-9s %4d %3d %3d %5.0f | %6.2f %6.2f %6.2f %6.2f | %6.2f %6.2f %+6.2f | %5.2f %+6.2f | %+.4f / %+.2f\n', ...
            r.id, r.maneuver, r.PEEP, r.dP, numel(r.kei), 60*r.RR, mq(1), iq(1), mq(2), iq(2), nn, ...
            mq(2)/max(mq(1), eps), r.real.lag, r.real.a, r.real.cc);
    S(end+1) = struct('subj', r.subject, 'PEEP', r.PEEP, 'tot', sum(mq), 'lag', r.real.lag); %#ok<SAGROW>
end
for subj = unique([S.subj])
    s = S([S.subj] == subj);
    if numel(s) < 4 || ~ismember(subj, SUBJ_RUN); continue; end
    [rs1, p1v] = spearman_perm([s.PEEP], [s.tot], 20000);
    [rs2, p2v] = spearman_perm([s.PEEP], [s.lag], 20000);
    fprintf('patient %d (n = %d recordings): Spearman PEEP vs total quadrature PV rho %+.2f (perm p %.3f) | PEEP vs lag index rho %+.2f (perm p %.3f)\n', ...
            subj, numel(s), rs1, p1v, rs2, p2v);
end
fprintf('H1: real >> null in every file with a null; H2: |rho| large with small p; H3: lag index > 0 at low PEEP.\n');

% ======================================================================
function [rho, p] = spearman_perm(x, y, nperm)
% Spearman rank correlation (average ranks for ties) and a two-sided
% permutation p-value
rx = tiedrank_(x(:));  ry = tiedrank_(y(:));
rho = corr_(rx, ry);
rs = RandStream('mt19937ar', 'Seed', 3);  cnt = 0;
for k = 1:nperm
    if abs(corr_(rx, ry(randperm(rs, numel(ry))))) >= abs(rho) - 1e-12; cnt = cnt + 1; end
end
p = (cnt + 1) / (nperm + 1);
end

function r = tiedrank_(x)
[xs, o] = sort(x);  r = zeros(size(x));  n = numel(x);  i = 1;
while i <= n
    j = i;  while j < n && xs(j+1) == xs(i); j = j + 1; end
    r(o(i:j)) = (i + j) / 2;  i = j + 1;
end
end

function c = corr_(a, b)
a = a - mean(a);  b = b - mean(b);  c = (a' * b) / max(norm(a) * norm(b), eps);
end

function q = prc(x, pc)
x = sort(x(isfinite(x)));  n = numel(x);  if n == 0; q = NaN; return; end
r = 1 + (n-1)*pc/100;  lo = floor(r);  hi = ceil(r);
q = x(lo) + (r - lo) * (x(hi) - x(lo));
end

function Es = phase_rand(E, seed)
rng(seed);
T = size(E, 2);  nh = floor((T-1)/2);
ph = exp(1i * 2*pi * rand(1, nh));
P  = ones(1, T);  P(2:nh+1) = ph;  P(T-nh+1:T) = conj(fliplr(ph));
Es = real(ifft(fft(E, [], 2) .* P, [], 2));
end

function v = p2p(x); v = max(x) - min(x); end

function [S, f] = spectrum(g, fps)
g = g(:)';  T = numel(g);
g = g - polyval(polyfit(1:T, g, 1), 1:T);
w = 0.5 - 0.5*cos(2*pi*(0:T-1)/(T-1));
nf = 8*2^nextpow2(T);
S = abs(fft(g .* w, nf));  f = (0:nf-1) / nf * fps;
keep = f <= fps/2;  S = S(keep);  f = f(keep);
end

function fpk = peak_in(S, f, band)
in = f >= band(1) & f <= band(2);  fi = f(in);  [~, k] = max(S(in));  fpk = fi(k);
end

function Y = fft_lowpass(X, fps, fc, tw)
T  = size(X, 2);
s  = (0:T-1) / (T-1);
tr = X(:,1) + (X(:,end) - X(:,1)) .* s;
f  = abs([0:floor(T/2), -ceil(T/2)+1:-1]) * fps / T;
H  = double(f <= fc - tw/2);
m  = f > fc - tw/2 & f < fc + tw/2;
H(m) = 0.5 * (1 + cos(pi * (f(m) - (fc - tw/2)) / tw));
Y  = real(ifft(fft(X - tr, [], 2) .* H, [], 2)) + tr;
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

function hp = lcurve_hp(A, Y)
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
end

function X = gn_frames(A, Y, hp)
p   = 1 ./ sqrt(sum(A.^2, 1))';
PAt = p .* A';
X   = PAt * ((A*PAt + hp^2*eye(size(A,1))) \ Y);
end

function B = insp_pv(q, kmin, kei, dt)
% PV per breath during inspiration, both directions, from a (T-1) x 1 rate
nb = numel(kei);  B = nan(nb, 2);
for b = 1:nb
    st = kmin(b):kei(b)-1;
    if numel(st) < 2; continue; end
    B(b,:) = [dt*sum(max(q(st), 0)), dt*sum(max(-q(st), 0))];
end
end

function TT = timing_thirds(q, kmin, kei, dt, TVg)
% PV in the early / mid / late third of each inspiration, both directions
nb = numel(kei);  X = nan(nb, 6);
for b = 1:nb
    st = kmin(b):kei(b)-1;  n = numel(st);
    if n < 3; continue; end
    u = ((0:n-1) + 0.5) / n;  bin = min(floor(3*u) + 1, 3);
    for h = 1:3
        qs = q(st(bin == h));
        X(b, h)   = 100 * dt * sum(max(qs, 0))  / TVg(b);
        X(b, 3+h) = 100 * dt * sum(max(-qs, 0)) / TVg(b);
    end
end
TT = median(X, 1, 'omitnan');
end

