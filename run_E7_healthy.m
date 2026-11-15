%RUN_E7_HEALTHY  E7: healthy adult with breath holds (Draeger PulmoVista 500,
%   20 fps) - an IN-VIVO null and a frame-rate check.
%
%   Data: EIDORS contrib "et-healthy-breathing" (E. Teschner, 2024; CC-ND,
%   attribution not required): healthy adult male, seated, belt at the 4th-5th
%   intercostal space; tidal breathing, a deeper breath, an expiratory and an
%   inspiratory breath hold; 20 frames/s, 60 s. Read from the Draeger *.eit
%   file (EIDORS: adjacent drive, 'rotate_meas').
%   Why: during a breath hold there is no gas flow, so ANY quadrature volume
%   there is a false positive of the whole chain (hardware, cardiac residue,
%   reconstruction, HK, readout) - a null measured in vivo, not simulated.
%   Seated: gravity is cranio-caudal, the A/P split is NOT gravity-dependent.
%
%   Model: adult_male_16el_lungs (cached; the EIDORS example uses the 32-el
%   model with every other electrode, i.e. the same positions within half an
%   electrode). Orientation: chosen by the cardiac map among 4 candidates
%   (e1 anterior or posterior x numbering towards left or right): heart
%   anterior and on the subject's left. First run (e1 anterior only): heart
%   POSTERIOR for both directions (-0.27 / -0.32) -> v2 also tries e1 posterior.
%   Domain: whole thorax (domain_graph 'thorax'), HK protocol P1, cardiac
%   low-pass as E3 v2.
%   Datasets (DS): draeger (done 2026-11-13, v2), swisstom (Swisstom BB2 2014,
%   28.1 fps, 32 el skip 4, adult_male_32el_lungs - netgen builds it the first
%   time), midas (Midas Ventom 2024, 19.8 fps, 16 el opposite drive). One
%   results file per dataset, E7_results_<name>.mat; done datasets are skipped.
%   Draeger runs (~3 h):
%     real   20 fps, whole record
%     null   20 fps, in-phase null (w * GN volume + phase-randomized residual)
%     real10 / real5 / real2p5: every 2nd / 4th / 8th frame of the low-passed
%            data - sampling check (frames per breath, see RUNS)
%   Per run: tidal breaths (not touching a hold) -> HK quadrature PV per
%   breath, image baselines (pend_estimators); holds -> quadrature volume in
%   inspiration-length windows tiled over each hold, in % of the median
%   tidal TV, for HK and for GN regional quadrature.
%   Self-check: GN global gas must follow the voltage sum (corr > 0.8).
%   swisstom / midas: real + null at ~10 fps only (~1 h each).
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clearvars; clc;
if ~exist('mk_library_model', 'file'); error('EIDORS not on path.'); end
here = fileparts(mfilename('fullpath'));  if isempty(here); here = pwd; end
P1 = struct('alpha', 1e-3, 'beta', 1e-2, 'dm', 4, 'kappa', 3);
sig_lung = 0.4;  N_lung = 800;  tau0 = 100;  it_hk = 6000;  hp_img = 0.1;
ORIENT = 'native';   % 'native': library electrode order, as in each dataset's EIDORS example (2026-11-13 v4);
                     % 'cardiac': choose among 8 rotations by the cardiac map. v3 showed the map is NOT a
                     % reliable cue with library models: the heart estimate does not rotate with the electrodes
                     % (Draeger magnitudes 0.09-0.35 across rotations) and the pick depends on the criterion.
HOLD_S = 4;  HOLD_TOL = 0.15;
MAPS_ONLY = {};  % e.g. {'draeger', 2; 'midas', 2}:   % non-empty: these datasets, 'real' at this decimation only, maps for
                                          % Fig16 -> maps\E7_<name>.mat ({} = full E7; swisstom excluded, see README)     % hold = >= 4 s with 2-s gas range < 15 % of the median tidal TV
% datasets (EIDORS contrib, healthy, seated, tidal breathing + expiratory and inspiratory holds).
% swisstom and midas: "the same subject as the GREIT chest model" (both pages) -> 2 subjects, 3 devices.
% E7 v2 (Draeger) showed PV unchanged down to ~19 frames/breath -> the others run at ~10 fps (HK cost).
DS = struct('name',   {'draeger', 'swisstom', 'midas'}, ...
            'file',   {fullfile('Draeger', 'ET_Test_File_Healthy_Lung_01.eit'), ...
                       fullfile('healthy_human', 'human-ventilation-2014.eit'), ...
                       fullfile('Midas_Medical_Ventom', 'EITData2024_10_31_08_43_09__S01.vtm')}, ...
            'model',  {'adult_male_16el_lungs', 'adult_male_32el_lungs', 'adult_male_16el_lungs'}, ...
            'reader', {'load', 'swisstom', 'load'}, ...
            'trim',   {0, 0, 100}, ...                      % midas: last 100 frames dropped, as the EIDORS example
            'runs',   {{'real', 1, false; 'null', 1, true; 'real10', 2, false; 'real5', 4, false; 'real2p5', 8, false}, ...
                       {'real', 3, false; 'null', 3, true}, ...      % 28.1 fps -> 9.4 fps
                       {'real', 2, false; 'null', 2, true}});        % 19.8 fps -> 9.9 fps
% frames per breath matter, not fps: neonate E2 has ~21 frames/breath; the Draeger adult (7.7 /min) has ~156 at
% 20 fps, ~39 at 5 fps and ~19 at 2.5 fps

if ~isempty(MAPS_ONLY)
    [~, im] = ismember(MAPS_ONLY(:,1), {DS.name});  DS = DS(im);
    for k = 1:numel(DS); DS(k).runs = {'real', MAPS_ONLY{k,2}, false}; end
    if ~exist(fullfile(here, 'maps'), 'dir'); mkdir(fullfile(here, 'maps')); end
end
for id = 1:numel(DS)
ds = DS(id);
ofile = fullfile(here, ['E7_results_' ds.name '.mat']);
if ~isempty(MAPS_ONLY); ofile = fullfile(here, 'maps', ['E7_' ds.name '.mat']); end
done = exist(ofile, 'file') == 2;     % cardiac mode: re-check the orientation, then skip
if done && strcmp(ORIENT, 'native'); fprintf('%s: done (skipped)\n', ds.name); continue; end
fprintf('\n################ %s ################\n', ds.name);
fn = fullfile(here, 'Data', ds.file);
% ---------------- data, rates, cardiac low-pass (as E3 v2) ----------------
msel = [];
switch ds.reader
    case 'load'
        D = load_real_eit(fn);  fps = D.fps;  V = real(D.vv);  stim = D.stim;
    case 'swisstom'                                  % EIDORS example: 32 el, skip 4, meas_select
        [vv, aux] = eidors_readdata(fn);  V = real(vv);  fps = 1 / (median(diff(aux.t_rel)) * 1e-6);
        [stim, msel] = mk_stim_patterns(32, 1, [0,5], [0,5], {'no_meas_current_next1'}, 1);
        if size(V,1) ~= numel(msel) && size(V,1) ~= nnz(msel)
            error('E7 swisstom: %d data rows, meas_select has %d (%d selected) - reader layout not as in the EIDORS example', ...
                  size(V,1), numel(msel), nnz(msel));
        end
        fprintf('swisstom: %d rows x %d frames, %.2f fps, meas_select %d of %d\n', size(V,1), size(V,2), fps, nnz(msel), numel(msel));
end
V = V(:, 1:end-ds.trim);
rsel = true(size(V,1), 1);  if ~isempty(msel) && size(V,1) == numel(msel); rsel = msel(:); end
g = sum(V(rsel,:) ./ mean(V(rsel,:), 2) - 1, 1);
[S_, frq] = spectrum(g, fps);
RR = peak_in(S_, frq, [0.08 1.0]);
hb = frq >= max(1.6*RR, 0.75) & frq <= min(3.5, 0.45*fps);
[pk, kh] = max(S_ .* hb);  HR = frq(kh);  prom = pk / median(S_(hb));
if prom >= 4 && 0.8*HR > 2.5*RR
    fc = min(0.8*HR, max(2.5*RR, 0.5*(2.5*RR + 0.8*HR)));
else
    HR = NaN;  fc = min(3*RR, 0.4*fps);
end
V_lp = fft_lowpass(V, fps, fc, 0.2*fc);
fprintf('rates: RR %.2f Hz (%.1f /min), HR %.2f Hz (%.0f /min, prominence %.1f) -> fc %.2f Hz | %.0f s\n', ...
        RR, 60*RR, HR, 60*HR, prom, fc, size(V,2)/fps);

% ---------------- model, orientation by the cardiac map ----------------
fm0 = mk_library_model(ds.model);
Ne = size(fm0.elems, 1);  nel = numel(fm0.electrode);
isL = false(Ne,1);  isL(fm0.mat_idx{2}) = true;  isR = false(Ne,1);  isR(fm0.mat_idx{3}) = true;
s0 = ones(Ne,1);  s0(isL | isR) = sig_lung;
ce = interp_mesh(fm0, 0);  vol = get_elem_volume(fm0);
[ant, rgt, c_t] = axes_dir(fm0);
if strcmp(ds.reader, 'swisstom')                 % measurement order: EIDORS example (rotate) vs reader (no_rotate)
    sopt = {{'no_meas_current_next1'}, {'no_meas_current_next1', 'no_rotate_meas'}};  mfs = nan(1, 2);
    for q = 1:2
        [stq, msq] = mk_stim_patterns(32, 1, [0,5], [0,5], sopt{q}, 1);
        fq = fm0;  fq.stimulation = stq;  fq.meas_select = msq;  fq = mdl_normalize(fq, 1);
        Jq = calc_jacobian(mk_image(fq, s0));  Yq = calc_difference_data(mean(V, 2), V(:, 1:10:end), fq);
        pq = 1 ./ sqrt(sum(Jq.^2, 1))';  Xq = (pq .* Jq') * ((Jq * (pq .* Jq') + hp_img^2*eye(size(Jq,1))) \ Yq);
        mfs(q) = norm(Jq*Xq - Yq, 'fro') / norm(Yq, 'fro');
    end
    [~, q] = min(mfs);  [stim, msel] = mk_stim_patterns(32, 1, [0,5], [0,5], sopt{q}, 1);
    fprintf('swisstom measurement order: GN misfit rotate_meas %.3f, no_rotate_meas %.3f -> %s\n', mfs, strjoin(sopt{q}, '+'));
end
% candidates: native order, or 8 rotations (e1 a quarter turn apart x numbering direction)
cands = [repmat({'left'; 'right'}, 4, 1), num2cell(kron((0:3)' * nel/4, [1; 1]))];   % numbering side, shift
if strcmp(ORIENT, 'native'); nca = 1; else; nca = size(cands, 1); end
labs = cell(1, nca);  ODS = cell(1, nca);
hcs = nan(nca, 2);  FM = cell(1, nca);
eang = @(e) atan2d(-(e * rgt'), e * ant');                         % electrode angle: 0 anterior, +90 subject's left
ee = nel/4 + 1;                                                     % the electrode a quarter turn from e1
for k = 1:nca
    if strcmp(ORIENT, 'native'); od = 1:nel;
    else; od = electrode_order(fm0, cands{k,1});  od = circshift(od, -cands{k,2}); end
    ODS{k} = od;
    fm = fm0;  fm.electrode = fm.electrode(od);
    epos = @(i) mean(fm0.nodes(fm0.electrode(od(i)).nodes, 1:2), 1) - c_t;
    labs{k} = sprintf('e1 at %+4.0f deg, e%d at %+4.0f deg (0 anterior, +90 left)', eang(epos(1)), ee, eang(epos(ee)));
    fm.stimulation = stim;  if ~isempty(msel); fm.meas_select = msel; end
    fm = mdl_normalize(fm, 1);
    J  = calc_jacobian(mk_image(fm, s0));
    FM{k} = struct('fm', fm, 'J', J);
    if isnan(HR); continue; end
    p = 1 ./ sqrt(sum(J.^2, 1))';  PJt = p .* J';
    Vc = fft_lowpass(V, fps, HR + 0.4, 0.2) - fft_lowpass(V, fps, max(HR - 0.4, fc + 0.1), 0.2);
    vh0 = mean(V, 2);
    xc = (PJt / (J*PJt + hp_img^2*eye(size(J,1)))) * calc_difference_data(vh0, vh0 + Vc, fm);
    gc = sum(Vc(rsel,:) ./ vh0(rsel), 1);  gc = gc - mean(gc);
    cr = ((xc - mean(xc, 2)) * gc') ./ (sqrt(sum((xc - mean(xc, 2)).^2, 2)) * norm(gc) + eps);
    sl = sign(sum(cr(isL | isR) .* vol(isL | isR)));
    cand = find(~(isL | isR) & cr * sl < -0.5);
    if isempty(cand); continue; end
    sco = max(-sl * cr(cand), 0) .* std(xc(cand,:), 0, 2);
    [~, o] = sort(sco, 'descend');  h = cand(o(1:max(1, ceil(0.1*numel(cand)))));
    hcs(k,:) = [dot(sum(ce(h,1:2) .* vol(h), 1)/sum(vol(h)) - c_t, ant), dot(sum(ce(h,1:2) .* vol(h), 1)/sum(vol(h)) - c_t, rgt)];
    fprintf('orientation %s: heart at anterior %+.3f, right %+.3f\n', labs{k}, hcs(k,:));
end
if strcmp(ORIENT, 'native') || all(isnan(hcs(:,1)))
    ko = 1;
else
    [~, ko] = max(hcs * [cosd(30); -sind(30)]);    % heart expected anterior-left: projection on 30 deg left of anterior
end
fm = FM{ko}.fm;  J = FM{ko}.J;
fprintf('chosen: %s\n', labs{ko});
if done                                           % cardiac mode only (native returned above)
    L0 = load(ofile, 'cands', 'ko');
    od0 = circshift(electrode_order(fm0, L0.cands{L0.ko,1}), -L0.cands{L0.ko,2});
    if isequal(od0, ODS{ko}); fprintf('%s: done, same orientation as the saved run (skipped)\n', ds.name);
    else; warning('E7:%s: saved run used ANOTHER orientation - rename %s to rerun it', ds.name, ofile); end
    continue
end
grp = 3 * ones(Ne,1);  grp(isL) = 1;  grp(isR) = 2;
Gc = domain_graph(fm, grp, N_lung, 'thorax');
Ac = J * Gc.EL;
ant_c = (Gc.cent(:,1:2) - c_t) * ant' > 0;
L_ap  = norm(sum(Gc.cent(ant_c,:).*Gc.vol(ant_c),1)/sum(Gc.vol(ant_c)) - sum(Gc.cent(~ant_c,:).*Gc.vol(~ant_c),1)/sum(Gc.vol(~ant_c)));
fprintf('model %s, whole thorax: %d clusters\n', ds.model, Gc.N);

% ---------------- runs ----------------
RUNS = ds.runs;
E7 = struct();
for ir = 1:size(RUNS, 1)
    t0 = tic;
    try
        [E7.(RUNS{ir,1}), MP] = analyse(V_lp(:, 1:RUNS{ir,2}:end), RUNS{ir,2}/fps, RR, RUNS{ir,3}, fm, J, Gc, Ac, ant_c, L_ap, ...
                                        P1, tau0, it_hk, HOLD_S, HOLD_TOL, g(1:RUNS{ir,2}:end));
    catch err
        if ~strcmp(err.identifier, 'E7:noBreaths'); rethrow(err); end
        fprintf('%s %s: %s\n', ds.name, RUNS{ir,1}, err.message);  E7 = [];  break
    end
    fprintf('%s done (%.0f min)\n', RUNS{ir,1}, toc(t0)/60);
end
if isempty(E7); continue; end
if ~isempty(MAPS_ONLY)
    MP.label = sprintf('Healthy adult, %s (seated, %.0f fps)', ds.name, fps / RUNS{1,2});
    save(ofile, 'MP', 'fm', 'Gc');  fprintf('maps saved %s (HK quadrature %.2f %%TV)\n', ofile, median(MP.PVq));
    continue
end
dsname = ds.name;
od = ODS{ko};
save(ofile, 'E7', 'RR', 'HR', 'fc', 'ko', 'hcs', 'cands', 'labs', 'RUNS', 'fps', 'dsname', 'od', 'ORIENT', 'stim', 'msel');
end

if ~isempty(MAPS_ONLY); return; end
% ---------------- summary over all datasets ----------------
fprintf('\n============ E7 healthy adults (seated): quadrature volume, %% of tidal volume ============\n');
fprintf('%-8s %-7s | %3s %3s | %-24s | %-24s | %-27s | %s\n', 'data', 'run', 'nb', 'nh', 'tidal HK quad (P>A A>P)', 'tidal GN opp qGN FRIC Sang', ...
        'HOLDS HK quad md / max', 'holds qGN md / max | lag ms');
for id = 1:numel(DS)
    ofile = fullfile(here, ['E7_results_' DS(id).name '.mat']);
    if ~exist(ofile, 'file'); continue; end
    L = load(ofile);
    for ir = 1:size(L.RUNS, 1)
        R = L.E7.(L.RUNS{ir,1});
        fprintf('%-8s %-7s | %3d %3d | %5.2f %5.2f (tot %5.2f) | %5.2f %5.2f %5.2f %6.2f | %6.2f / %6.2f (n %3d)  | %5.2f / %5.2f | %+.0f\n', ...
                DS(id).name, L.RUNS{ir,1}, numel(R.valid), size(R.holds, 1), median(R.Bq(R.valid,:), 1, 'omitnan'), ...
                sum(median(R.Bq(R.valid,:), 1, 'omitnan')), median(R.Bb(R.valid,[1 2 3 5]), 1, 'omitnan'), ...
                median(R.holdHK), max(R.holdHK), numel(R.holdHK), median(R.holdGN), max(R.holdGN), R.lag);
    end
end
fprintf(['Reading: HOLDS = in-vivo false positive of the whole chain (no flow); compare with the tidal values and\n' ...
         'with the simulated / null floors. Decimated runs: effect of frames per breath on PV and lag.\n']);

% ======================================================================
function [R, MP] = analyse(V, dt, RR, isnull, fm, J, Gc, Ac, ant_c, L_ap, P1, tau0, it_hk, HOLD_S, HOLD_TOL, gsum)
T = size(V, 2);  fps = 1/dt;
minsep = round(min(0.6 / RR, 3) * fps);   % cap 3 s: holds make the spectral RR too low (swisstom 0.08 Hz, true ~0.17)
vh = mean(V, 2);
for pass = 1:2
    Y   = calc_difference_data(vh, V, fm);
    hp  = lcurve_hp(Ac, Y);
    rgn = -gn_frames(Ac, Y, hp);
    gV  = Gc.vol' * rgn;
    kmin = breath_minima(gV, minsep);
    vh  = mean(V(:, kmin), 2);
end
cg = corrcoef(gV, gsum);
assert(cg(1,2) > 0.8, 'E7 self-check: GN gas vs voltage sum corr %.2f (sign or reader problem)', cg(1,2));
nb = numel(kmin) - 1;  kmin = kmin(:)';
kei = zeros(1, nb);
for b = 1:nb; kk = kmin(b):kmin(b+1);  [~, ip] = max(gV(kk));  kei(b) = kk(ip); end
TVg = arrayfun(@(b) gV(kei(b)) - gV(kmin(b)), (1:nb)');
% holds: >= HOLD_S s whose 2-s gas range is < HOLD_TOL x median TV
w2 = max(3, round(2*fps));
rngw = movmax(gV, w2) - movmin(gV, w2);
hm = rngw < HOLD_TOL * median(TVg);
d = diff([0, hm, 0]);  hs = find(d == 1);  he = find(d == -1) - 1;
keep = (he - hs + 1) >= round(HOLD_S * fps);  hs = hs(keep);  he = he(keep);
holds = [hs(:), he(:)];
inhold = false(1, T);  for h = 1:size(holds,1); inhold(holds(h,1):holds(h,2)) = true; end
valid = find(arrayfun(@(b) ~any(inhold(kmin(b):kmin(b+1))), 1:nb));
fprintf('  end-expirations at %s s\n', mat2str(round((kmin - 1) * dt, 1)));
if isempty(valid)
    error('E7:noBreaths', 'no complete tidal breath outside the holds - dataset skipped');
end
fprintf('  %.0f fps: %d breaths (%d not touching a hold), holds %s s | GN hp %.3g | corr(GN gas, voltage sum) %+.2f\n', ...
        fps, nb, numel(valid), mat2str(round(holds' * dt, 1)), hp, cg(1,2));
% HK
w = max(mean(rgn(:, kei(valid)) - rgn(:, kmin(valid)), 2), 0);
rho0 = P1.kappa * max(sum(Gc.vol .* w) / sum(Gc.vol), 0);
o = struct('alpha', P1.alpha, 'beta', P1.beta, 'delta', P1.dm*L_ap/pi, 'gamma', 1e3, 'dt', dt, 'max_iter', it_hk, ...
           'tol', 1e-4, 'verbose', 0, 'use_eta', false, 'normalize', true, 'tau0', tau0, 'rho0', rho0, 'vent_w', w);
if isnull; Yk = -Ac * (w * (gV / (Gc.vol' * w))) + phase_rand(Y + Ac*rgn, 1);  rgk = -gn_frames(Ac, Yk, hp);
else;      Yk = Y;  rgk = rgn; end
[rho, m, ~, info] = hk_core(-J, speye(size(J,1)), Yk, Gc, o);
QG = hk_pendelluft(Gc, m, ant_c, dt);
Fl = (diff(Gc.vol' * rho) / dt)';
a  = (Fl' * QG) / (Fl' * Fl);  r = QG - a * Fl;
gk = Gc.vol' * rgk;  TVk = arrayfun(@(b) gk(kei(b)) - gk(kmin(b)), (1:nb)');
Bq = 100 * insp_pv(r, kmin, kei, dt) ./ TVk;
[Bb, lag] = pend_estimators(rgk, Gc.vol, ant_c, kmin, kei, dt);
% holds: inspiration-length windows
Wn = max(2, round(median(kei(valid) - kmin(valid))));
TVm = median(TVk(valid));
dS = diff(Gc.vol(ant_c)' * rgk(ant_c,:));  dG = diff(gk);  rG = dS - (dG*dS')/(dG*dG') * dG;
holdHK = [];  holdGN = [];
for h = 1:size(holds, 1)
    s0 = holds(h,1) + round(0.5*fps);  s1 = holds(h,2) - round(0.5*fps);
    for st = s0 : Wn : s1 - Wn + 1
        ix = st : st + Wn - 2;
        holdHK(end+1) = 100 * dt * sum(abs(r(ix))) / TVm; %#ok<AGROW>
        holdGN(end+1) = 100 * sum(abs(rG(ix))) / TVm;      %#ok<AGROW>
    end
end
R = struct('kmin', kmin, 'kei', kei, 'valid', valid, 'holds', holds, 'TV', TVk, 'Bq', Bq, 'Bb', Bb, 'lag', lag, ...
           'holdHK', holdHK, 'holdGN', holdGN, 'a', a, 'Wn', Wn, 'hp', hp, ...
           'mis', norm(-Ac*rho - Yk, 'fro')/norm(Yk, 'fro'), 'iters', info.iters, 'res', max(info.res), 'gV', gV);
MP = [];
if nargout > 1; MP = pend_maps(Gc, rgk, rho, m, kmin(valid), kei(valid), ant_c, dt); end   % tidal breaths only
fprintf('  HK: misfit %.3f, %d it | tidal quad P>A %.2f A>P %.2f %%TV | holds HK md %.2f max %.2f, GN md %.2f (%d windows of %d frames)\n', ...
        R.mis, info.iters, median(Bq(valid,:), 1, 'omitnan'), median(holdHK), max(holdHK), median(holdGN), numel(holdHK), Wn);
end

function kmin = breath_minima(gV, minsep)
% end-expiration frames. Mirror padding by minsep makes a minimum at the
% record edge detectable (swisstom: first tidal breath starts at ~0.5 s);
% prominence > 10 % of the range (v2 Draeger used 15 % without padding).
% shortcut: a relative prominence still fails if deep breaths dwarf the tidal ones - then set a tidal window per dataset
T  = numel(gV);  P = min(minsep, T - 1);
gp = [fliplr(gV(2:P+1)), gV, fliplr(gV(T-P:T-1))];
k  = find(islocalmin(gp, 'MinSeparation', minsep, 'MinProminence', 0.10*(max(gV) - min(gV)))) - P;
kmin = k(k >= 1 & k <= T);
end

function Es = phase_rand(E, seed)
rng(seed);
T = size(E, 2);  nh = floor((T-1)/2);
ph = exp(1i * 2*pi * rand(1, nh));
P  = ones(1, T);  P(2:nh+1) = ph;  P(T-nh+1:T) = conj(fliplr(ph));
Es = real(ifft(fft(E, [], 2) .* P, [], 2));
end

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
nb = numel(kei);  B = nan(nb, 2);
for b = 1:nb
    st = kmin(b):kei(b)-1;
    if numel(st) < 2; continue; end
    B(b,:) = [dt*sum(max(q(st), 0)), dt*sum(max(-q(st), 0))];
end
end
