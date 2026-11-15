%PREP_E1  Data preparation and sanity checks for experiment E1.  (v2)
%
%   Healthy volunteer, Viasys/Goettingen, Gomez-Laberge 2005 (EIDORS
%   contrib "cg_normal_breathing"). Recordings used (README_1pl/_2pl):
%     _001 tidal breathing, rest           _003 paradoxical breathing, rest
%     _004 tidal breathing, post-exercise  _006 paradoxical, post-exercise
%   Paradoxical: breath held, air moved rib cage <-> abdomen ("repeated
%   every 5 seconds"). With the glottis closed the total gas volume is
%   ~constant -> physiological check of the HK global source (E1 pred. b).
%
%   NO HK reconstruction here. This script
%     1. loads + QCs every recording (load_real_eit: 208 measured rows),
%     2. VERIFIES the frame rate (no .prl for this set): candidate fps
%        {10,13,20,25,44,50} are scored by the cardiac fundamental + 2
%        harmonics at the README heart rate; then reports the paradoxical
%        cycle frequency at the chosen fps (5 s or 10 s period?),
%     3. removes the cardiac component (zero-phase FFT low-pass, core MATLAB),
%        builds reference frames,
%     4. builds the 1-plane library model and the 2-plane model: the EIDORS
%        2x16 library model (32 electrodes) restricted to the 16 electrodes
%        of the 8+8 zig-zag layout (odd positions on one ring, even on the
%        other; selected by measured angle and height), with
%        orientation checks: lung labels (right lung larger), e1 anterior
%        (rotates the numbering if not), e5 on the subject's right (flips if
%        not), and the cranial direction of +z (S = R x A) for the planes,
%     5. GN one-step SANITY images (Wiener form - no dense Ne x Ne matrix):
%        paradoxical cycle, and a sign-aware cardiac map (heart = non-lung
%        elements pulsing opposite to the lungs; must be anterior-left),
%     6. saves E1_prep.mat (next to this file) for the later HK run.
%
%   Needs EIDORS on the path. The 2x16 library model is built by netgen
%   once (EIDORS model store). Run the whole file (not Run Section).
%
%   Author:  Hoang Nhut Huynh, National Chung Hsing University, Taichung, Taiwan
%   Project: HK-dEIT, Hellinger-Kantorovich regularized dynamic EIT for pendelluft
%            quantification (see README.md for the experiment map)
%   Core MATLAB (R2017b or later) and EIDORS v3.12 where noted; no toolboxes.

clear; clc;
if ~exist('eidors_readdata', 'file')
    error('EIDORS not on path - run its startup.m first.');
end

% ---------------- settings ----------------
here = fileparts(mfilename('fullpath'));
if isempty(here); here = pwd; end              % Run Section / Evaluate Selection
data_root   = fullfile(here, 'Data', 'normal_human');
fps_assumed = 13;            % Viasys default (no .prl in this data set)
fps_cands   = [10 13 20 25 44 50];
sig_lung    = 0.4;           % Jacobian background, same as E4
hp_gn       = 0.03;          % GN one-step hyperparameter - SANITY images only
z_planes    = [0.65 0.35];   % [5th-rib, 7th-rib] electrode planes if +z is
                             % cranial (EIDORS 2x16 library heights); swapped
                             % automatically if +z turns out to be caudal
maxh        = 0.08;                            % library mesh size (for reports)

recs = struct( ...
    'plane', {1, 1, 1, 1, 2, 2, 2, 2}, ...
    'id',    {'001','003','004','006','001','003','004','006'}, ...
    'kind',  {'tidal','paradox','tidal','paradox','tidal','paradox','tidal','paradox'}, ...
    'state', {'rest','rest','exercise','exercise','rest','rest','exercise','exercise'}, ...
    'HR',    {62, 62, 92, 84, 58, 60, 102, 76});      % bpm, from the READMEs

% ---------------- 1-3: load, verify fps, filter, reference -------------
fprintf('\n%-8s %6s | %-34s | %-8s | %s\n', 'record', 'frames', ...
        sprintf('cardiac comb score, fps = %s', mat2str(fps_cands)), 'best', 'QC');
for i = 1:numel(recs)
    r = recs(i);
    fn = fullfile(data_root, sprintf('cg_invivo_%dpl', r.plane), ...
                  sprintf('cgomez-%dpl', r.plane), ...
                  sprintf('vivo150905_t%d_%s.get', r.plane, r.id));
    w0 = warning('off', 'load_real_eit:fps');          % expected; checked below
    D  = load_real_eit(fn, struct('fps', fps_assumed));
    warning(w0);
    V  = D.vv;
    g  = sum(V ./ mean(V, 2) - 1, 1);                  % normalized global signal

    [S, fcpf] = spectrum(g);
    sc = zeros(size(fps_cands));
    for c = 1:numel(fps_cands)
        sc(c) = comb_score(S, fcpf, r.HR/60/fps_cands(c));
    end
    [~, cb] = max(sc);

    recs(i).file   = fn;
    recs(i).V      = V;
    recs(i).g      = g;
    recs(i).stim   = D.stim;
    recs(i).qc     = D.qc;
    recs(i).fps_scores = sc;
    recs(i).fps_best   = fps_cands(cb);

    qcs = sprintf('bad %d, chan %d, jump %d', numel(D.qc.bad_frames), ...
                  numel(D.qc.chan_flag), numel(D.qc.jump_frames));
    fprintf('%dpl_%s %6d | %-34s | %-8g | %s\n', r.plane, r.id, D.nframes, ...
            sprintf('%5.1f ', sc), fps_cands(cb), qcs);
end
fps_vote = mode([recs.fps_best]);
fprintf(['(score = sum over k=1..3 of w_k * peak/median near k*HR, w = 1, .5, .25;\n' ...
         ' a clear winner should beat the others by > 2x.)  -> majority fps = %g\n'], fps_vote);
if fps_vote ~= fps_assumed
    warning('prep_E1:fps', ['Data favour %g fps, not the assumed %g. ' ...
            'Set fps_assumed = %g and re-run.'], fps_vote, fps_assumed, fps_vote);
end
fps = fps_assumed;

for i = 1:numel(recs)
    r  = recs(i);
    fh = r.HR/60;
    % paradoxical cycle at the chosen fps: dominant frequency in 0.05-0.5 Hz
    if strcmp(r.kind, 'paradox')
        [S, fcpf] = spectrum(r.g);
        in = fcpf*fps >= 0.05 & fcpf*fps <= 0.5;
        fi = fcpf(in);  [~, k] = max(S(in));
        recs(i).f_cycle = fi(k)*fps;
        fprintf('%dpl_%s paradoxical cycle: %.3f Hz (period %.1f s; README: "every 5 s")\n', ...
                r.plane, r.id, recs(i).f_cycle, 1/recs(i).f_cycle);
    else
        recs(i).f_cycle = NaN;
    end
    % cardiac removal: low-pass at 0.6*HR (keeps a 0.2 Hz cycle and its 2nd
    % harmonic; the 3rd is partly attenuated)
    fc   = 0.6*fh;
    V_lp = fft_lowpass(r.V, fps, fc, 0.2*fc);
    g_lp = sum(V_lp ./ mean(V_lp, 2) - 1, 1);
    % references: whole-record mean (paradox: volume ~constant) and, for
    % tidal breathing, the 10% frames with the lowest global signal (check on
    % the figure that inspiration INCREASES g, else use the highest 10%)
    [~, o] = sort(g_lp);
    n10    = max(5, round(0.1*numel(g_lp)));
    recs(i).V_lp    = V_lp;
    recs(i).g_lp    = g_lp;
    recs(i).vh_mean = mean(V_lp, 2);
    recs(i).vh_low  = mean(V_lp(:, o(1:n10)), 2);
    recs(i).fc      = fc;
end

% ---------------- 4: models + orientation ------------------------------
fmdl1 = mk_library_model('adult_male_16el_lungs');
[order, ori] = electrode_orientation(fmdl1);
fprintf('\nmodel: thorax centre (%+.3f, %+.3f) | anterior (%+.2f, %+.2f) | subject right (%+.2f, %+.2f)\n', ...
        ori.center, ori.ant, ori.right);
fprintf('  lung areas: left %.4f, right %.4f (expect right > left)\n', ori.vol_left, ori.vol_right);
fprintf('  library e1 at %+.1f deg from anterior; numbering order used: %s\n', ...
        ori.e1_offset_deg, mat2str(order));
fprintf('  E4 note: anterior has %s y -> E4''s "post = y < median" is %s\n', ...
        ternary(ori.ant(2) > 0, '+', '-'), ternary(ori.ant(2) > 0, 'POSTERIOR (ok)', 'ANTERIOR (labels swapped)'));
if ori.vol_right <= ori.vol_left
    warning('prep_E1:lungs', 'Right lung not larger than left: lung labels may be swapped - check before E1.');
end
if abs(ori.e1_offset_deg) > 15
    warning('prep_E1:e1', 'Library electrode closest to anterior is %.1f deg off - ring offset.', ori.e1_offset_deg);
end
fmdl1.electrode   = fmdl1.electrode(order);
fmdl1.stimulation = recs(1).stim;
fmdl1             = mdl_normalize(fmdl1, 1);   % in vivo: normalized difference

% cranial direction: superior = right x anterior (right-handed body frame)
sup_z = ori.right(1)*ori.ant(2) - ori.right(2)*ori.ant(1);
if sup_z < 0
    z_planes = fliplr(z_planes);
    fprintf('  +z is CAUDAL in this model -> 5th-rib (odd) electrodes placed at z = %.2f\n', z_planes(1));
else
    fprintf('  +z is cranial -> 5th-rib (odd) electrodes at z = %.2f\n', z_planes(1));
end

ang = ori.angle_deg(order);                     % data electrode k -> angle from centroid
zk  = z_planes(2 - mod(1:16, 2));               % odd -> 5th rib, even -> 7th rib
[fmdl2, sel2] = select_2plane_from_library(ang, zk);
fprintf('2-plane model: library 2x16 electrodes kept %s | max angle error %.2f deg, max z error %.3f\n', ...
        mat2str(sel2.sel'), sel2.max_dang, sel2.max_dz);
fmdl2.stimulation = recs(5).stim;
fmdl2             = mdl_normalize(fmdl2, 1);
ep1 = elec_centroids(fmdl1);  ep2 = elec_centroids(fmdl2);
fprintf('2-plane model: %d elements | max xy gap to 1-plane electrode k: %.3g (should be ~mesh size %.2g)\n', ...
        size(fmdl2.elems,1), max(sqrt(sum((ep2(:,1:2) - ep1(:,1:2)).^2, 2))), maxh);
fmdls = {fmdl1, fmdl2};

% ---------------- 5: GN one-step sanity images (Wiener form) ------------
lev = {[inf inf 0.5], [inf inf 0.65; inf inf 0.35]};
for pl = 1:2
    fm  = fmdls{pl};
    RM  = gn_wiener(fm, sig_lung, hp_gn);       % Ne x M
    lung = [fm.mat_idx{2}(:); fm.mat_idx{3}(:)];
    ce   = interp_mesh(fm, 0);
    vol  = get_elem_volume(fm);

    % (a) paradoxical cycle: mean of the 5% highest minus 5% lowest frames
    r  = recs([recs.plane] == pl & strcmp({recs.kind}, 'paradox') & strcmp({recs.state}, 'rest'));
    [~, o] = sort(r.g_lp);  n5 = max(5, round(0.05*numel(o)));
    vi = [mean(r.V_lp(:, o(end-n5+1:end)), 2), mean(r.V_lp(:, o(1:n5)), 2)];
    x  = RM * calc_difference_data(r.vh_mean, vi, fm);
    im = mk_image(fm, x(:,1) - x(:,2));
    if pl == 2
        cran = (ce(lung,3) - 0.5) * sign(sup_z) > 0;            % cranial half
        vw   = @(ix) sum(im.elem_data(ix).*vol(ix)) / sum(vol(ix));
        fprintf('2pl paradox (high-low g): lung change cranial half %+.3e, caudal half %+.3e\n', ...
                vw(lung(cran)), vw(lung(~cran)));
    end
    figure('Name', sprintf('E1 %dpl paradox', pl));
    show_slices(im, lev{pl});
    title(sprintf('%d-plane, paradoxical (rest): high - low global signal', pl));

    % (b) sign-aware cardiac map from tidal rest (band HR +/- 0.25 Hz)
    r  = recs([recs.plane] == pl & strcmp({recs.kind}, 'tidal') & strcmp({recs.state}, 'rest'));
    fh = r.HR/60;
    Vc = fft_bandpass(r.V, fps, fh - 0.25, fh + 0.25, 0.1);
    gc = sum(Vc ./ r.vh_mean, 1);                               % global cardiac signal
    xc = RM * calc_difference_data(r.vh_mean, r.vh_mean + Vc, fm);
    Xc = xc - mean(xc, 2);  gcc = gc - mean(gc);
    cr = (Xc * gcc') ./ (sqrt(sum(Xc.^2, 2)) * norm(gcc) + eps); % corr per element
    sl = sign(sum(cr(lung) .* vol(lung)));                      % lung pulse sign
    nl = setdiff((1:size(fm.elems,1))', lung);
    % v2: only elements in a thin slab around the electrode plane(s), where
    % the data are sensitive (far from the planes the 3-D image is a smooth
    % extrapolation and pulls the centroid to the middle); among the
    % non-lung elements pulsing opposite to the lungs keep the strongest 10%
    % by (anti-correlation x amplitude), weighted centroid.
    zc   = lev{pl}(:,3)';
    slab = any(abs(ce(:,3) - zc) < 0.1, 2);
    amp  = std(xc, 0, 2);
    sco  = max(-sl * cr, 0) .* amp;
    cand = nl(slab(nl) & cr(nl) * sl < -0.5);
    if isempty(cand)
        fprintf('%dpl cardiac map: no non-lung slab element pulsing opposite to the lungs\n', pl);
    else
        [~, oc] = sort(sco(cand), 'descend');
        heart = cand(oc(1:max(1, ceil(0.1*numel(cand)))));
        wt = vol(heart) .* sco(heart);
        hc = sum(ce(heart,1:2) .* wt, 1) / sum(wt) - ori.center;
        sx = sqrt(sum(wt .* sum((ce(heart,1:2) - ori.center - hc).^2, 2)) / sum(wt));
        fprintf(['%dpl cardiac map: heart (%d of %d slab candidates) at anterior %+.3f, right %+.3f, ' ...
                 'spread %.3f (expect anterior > 0, right < 0)\n'], pl, numel(heart), numel(cand), ...
                 dot(hc, ori.ant), dot(hc, ori.right), sx);
    end
    imc = mk_image(fm, cr);
    figure('Name', sprintf('E1 %dpl cardiac', pl));
    show_slices(imc, lev{pl});
    title(sprintf('%d-plane, correlation with global cardiac signal (tidal rest)', pl));
end

% ---------------- figure: global signals --------------------------------
figure('Name', 'E1 global signals');
for i = 1:numel(recs)
    subplot(4, 2, i);
    t = (0:numel(recs(i).g)-1) / fps;
    plot(t, recs(i).g, t, recs(i).g_lp);
    title(sprintf('%dpl %s %s (%s)', recs(i).plane, recs(i).id, recs(i).kind, recs(i).state));
    xlabel(sprintf('t [s] (%g fps)', fps));
end
legend('raw', 'cardiac-filtered');

% ---------------- 6: save ------------------------------------------------
recs_save = rmfield(recs, 'stim');                % stim lives in the models
save(fullfile(here, 'E1_prep.mat'), 'recs_save', 'fmdl1', 'fmdl2', 'order', 'ori', ...
     'fps', 'fps_vote', 'sig_lung', 'z_planes', 'sup_z');
fprintf('\nsaved %s\n', fullfile(here, 'E1_prep.mat'));

% ======================================================================
function [S, f] = spectrum(g)
% Hann-windowed, 8x zero-padded amplitude spectrum of a detrended signal;
% f in cycles/frame
g  = g(:)';
T  = numel(g);
g  = g - polyval(polyfit(1:T, g, 1), 1:T);
w  = 0.5 - 0.5*cos(2*pi*(0:T-1)/(T-1));
nf = 8*2^nextpow2(T);
S  = abs(fft(g .* w, nf));
f  = (0:nf-1) / nf;
keep = f <= 0.5;
S = S(keep);  f = f(keep);
end

function s = comb_score(S, f, f1)
% evidence that f1 (cycles/frame) is the cardiac FUNDAMENTAL: weighted sum
% over harmonics k = 1..3 of (local peak within +/-8%) / (median of the
% spectrum in 0.5..1.5 x k*f1). Harmonics beyond Nyquist are skipped.
w = [1 0.5 0.25];
s = 0;
for k = 1:3
    fk = k*f1;
    if fk*1.08 > 0.5; continue; end
    pk  = max(S(f >= 0.92*fk & f <= 1.08*fk));
    ref = median(S(f >= 0.5*fk & f <= min(0.5, 1.5*fk)));
    s   = s + w(k) * pk / (ref + eps);
end
end

function Y = fft_lowpass(X, fps, fc, tw)
% zero-phase low-pass along dim 2 (FFT mask, raised-cosine transition of
% width tw centred at fc, Hz; fc is the half-amplitude point). The line
% through the first and last sample is removed first and added back.
T  = size(X, 2);
s  = (0:T-1) / (T-1);
tr = X(:,1) + (X(:,end) - X(:,1)) .* s;
f  = abs([0:floor(T/2), -ceil(T/2)+1:-1]) * fps / T;
H  = double(f <= fc - tw/2);
m  = f > fc - tw/2 & f < fc + tw/2;
H(m) = 0.5 * (1 + cos(pi * (f(m) - (fc - tw/2)) / tw));
Y  = real(ifft(fft(X - tr, [], 2) .* H, [], 2)) + tr;
end

function Y = fft_bandpass(X, fps, fl, fh, tw)
Y = fft_lowpass(X, fps, fh, tw) - fft_lowpass(X, fps, fl, tw);
end

function RM = gn_wiener(fm, sig_lung, hp)
% GN one-step reconstruction matrix with a DIAGONAL prior (NOSER) in the
% Wiener form RM = P J' (J P J' + hp^2 I)^-1,  P = inv(RtR), W = I -
% identical to (J'J + hp^2 RtR) \ J' but only an M x M solve (no dense
% Ne x Ne matrix: ~5 GB for 26k elements).
s0 = ones(size(fm.elems,1), 1);
s0([fm.mat_idx{2}(:); fm.mat_idx{3}(:)]) = sig_lung;
imdl = eidors_obj('inv_model', 'E1 GN sanity');
imdl.fwd_model    = fm;
imdl.reconst_type = 'difference';
imdl.RtR_prior    = @prior_noser;
imdl.hyperparameter.value = hp;
imdl.jacobian_bkgnd.value = s0;
J   = calc_jacobian(mk_image(fm, s0));          % normalized if fm is
RtR = calc_RtR_prior(imdl);
p   = 1 ./ full(diag(RtR));
PJt = p .* J';                                  % Ne x M
RM  = PJt / (J*PJt + hp^2*eye(size(J,1)));
end

function [fm, info] = select_2plane_from_library(ang, zk)
% The EIDORS library model 'adult_male_2x16el_lungs' has 16 electrodes on
% each of two planes (z = 0.65, 0.35) at the SAME angular positions as the
% 1x16 model (equal-distance spacing). The 8+8 zig-zag layout of this data
% set (README_2pl) is the odd positions of one ring plus the even positions
% of the other, so we keep exactly those 16 electrodes; the other 16 become
% plain insulating boundary. (v1 meshed a custom 16-electrode geometry with
% ng_mk_extruded_model; netgen failed with "Problem in edge detection".)
% Selection is by measured centroid angle (from the volume-weighted thorax
% centre, as for the 1-plane model) and z - no assumption on the library's
% electrode numbering.
%   ang : 16 x 1 target angles [deg] of data electrodes 1..16
%   zk  : 16 x 1 target plane heights
fm  = mk_library_model('adult_male_2x16el_lungs');
ce  = interp_mesh(fm, 0);
vol = get_elem_volume(fm);
c_t = sum(ce(:,1:2) .* vol, 1) / sum(vol);
ep  = elec_centroids(fm);
ae  = atan2(ep(:,2) - c_t(2), ep(:,1) - c_t(1)) * 180/pi;
sel = zeros(16,1);  dang = zeros(16,1);  dz = zeros(16,1);
for k = 1:16
    cand = find(abs(ep(:,3) - zk(k)) < 0.1);
    if isempty(cand); error('select_2plane: no library electrode near z = %.2f', zk(k)); end
    d = abs(mod(ae(cand) - ang(k) + 180, 360) - 180);
    [dang(k), j] = min(d);
    sel(k) = cand(j);
    dz(k)  = abs(ep(sel(k),3) - zk(k));
end
if numel(unique(sel)) < 16
    error('select_2plane: two data electrodes mapped to the same library electrode');
end
if max(dang) > 5
    error('select_2plane: angle mismatch %.1f deg > 5 deg - library positions differ from the 1x16 model', max(dang));
end
fm.electrode = fm.electrode(sel);
info = struct('sel', sel, 'max_dang', max(dang), 'max_dz', max(dz));
end

function ep = elec_centroids(fm)
ne = numel(fm.electrode);
ep = zeros(ne, size(fm.nodes, 2));
for k = 1:ne
    ep(k,:) = mean(fm.nodes(fm.electrode(k).nodes, :), 1);
end
end

function [order, ori] = electrode_orientation(fm)
% README convention: e1 on the sternum, numbering towards the subject's
% RIGHT (e5 right axilla, e9 spine, e13 left axilla). Model geometry:
% mat_idx{2} = left lung, mat_idx{3} = right lung (shape order given to
% ng_mk_extruded_model); the lungs lie dorsal of the heart, so (thorax
% centre - lung centre) points ANTERIOR. Anterior is orthogonalized to
% the left->right axis. The numbering is ROTATED so that the electrode
% closest to anterior is e1, then FLIPPED if e5 is not on the right.
ce  = interp_mesh(fm, 0);
vol = get_elem_volume(fm);
wc  = @(ix) sum(ce(ix,1:2) .* vol(ix), 1) / sum(vol(ix));
c_t = wc(1:size(fm.elems,1));
c_l = wc(fm.mat_idx{2});   c_r = wc(fm.mat_idx{3});
rgt = c_r - c_l;            rgt = rgt / norm(rgt);
ant = c_t - (c_l + c_r)/2;  ant = ant - dot(ant, rgt)*rgt;  ant = ant / norm(ant);
ep  = elec_centroids(fm);
ep  = ep(:,1:2) - c_t;
a_from_ant = atan2(ep*rgt', ep*ant') * 180/pi;  % 0 = anterior, +90 = right
[~, k0] = min(abs(a_from_ant));
n  = size(ep, 1);
order = circshift(1:n, -(k0-1));
if dot(ep(order(5),:), rgt) <= 0
    order = order([1, n:-1:2]);
end
ori.center = c_t;  ori.ant = ant;  ori.right = rgt;
ori.vol_left  = sum(vol(fm.mat_idx{2}));
ori.vol_right = sum(vol(fm.mat_idx{3}));
ori.e1_offset_deg = a_from_ant(order(1));
ori.angle_deg = atan2(ep(:,2), ep(:,1)) * 180/pi;    % from the centroid, as
                                                     % ng_mk_extruded_model expects
end

function s = ternary(c, a, b)
if c; s = a; else; s = b; end
end
