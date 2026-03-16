mode(0);
ieee(1);

outdir = getenv("EX_LAG_MULT_OUTDIR");

A = [1 1 1];
E = [64 64 64];
L = [4 4 6];
alfa = [acos(3 / 4) -acos(3 / 4) 0];

Dvec = [1 2 5 6
        5 6 3 4
        1 2 3 4];

N = max(max(Dvec));
nc = 3;
I = eye(N, N);
S1 = I(Dvec(1, :), :);
S2 = I(Dvec(2, :), :);
S3 = I(Dvec(3, :), :);

kref = [1 0 -1 0
        0 0 0 0
       -1 0 1 0
        0 0 0 0];

c = cos(alfa);
s = sin(alfa);

T1 = [c(1) s(1) 0 0
     -s(1) c(1) 0 0
      0 0 c(1) s(1)
      0 0 -s(1) c(1)];
T2 = [c(2) s(2) 0 0
     -s(2) c(2) 0 0
      0 0 c(2) s(2)
      0 0 -s(2) c(2)];
T3 = [c(3) s(3) 0 0
     -s(3) c(3) 0 0
      0 0 c(3) s(3)
      0 0 -s(3) c(3)];

k1 = A(1) * E(1) / L(1) * kref;
k2 = A(2) * E(2) / L(2) * kref;
k3 = A(3) * E(3) / L(3) * kref;

kg1 = T1' * k1 * T1;
kg2 = T2' * k2 * T2;
kg3 = T3' * k3 * T3;

K = S1' * kg1 * S1 + S2' * kg2 * S2 + S3' * kg3 * S3;

G = [1 0 0 0 0 0
     0 1 0 0 0 0
     0 0 -sin(60 / 180 * %pi) cos(60 / 180 * %pi) 0 0];
Q = zeros(nc, 1);
a_G = max(max(K));
Gbar = a_G * G;
Qbar = a_G * Q;

P = zeros(N, 1);
P(6) = -10;

AL = [K Gbar'
      Gbar zeros(nc, nc)];
solution = AL \ [P; Qbar];

Lag = solution(N + 1:$) * a_G;
U = solution(1:N);
R = K([1 2 3 4], :) * U;

ueg1 = U(Dvec(1, :));
ueg2 = U(Dvec(2, :));
ueg3 = U(Dvec(3, :));

ue1 = T1 * ueg1;
ue2 = T2 * ueg2;
ue3 = T3 * ueg3;

local_displacements = [ue1 ue2 ue3];
member_forces = [k1 * ue1 k2 * ue2 k3 * ue3];
constraint_residual = G * U - Q;

if outdir <> "" then
    if ~isdir(outdir) then
        mkdir(outdir);
    end
    csvWrite(U, outdir + "/U.tsv", ascii(9));
    csvWrite(Lag, outdir + "/Lag.tsv", ascii(9));
    csvWrite(R, outdir + "/R.tsv", ascii(9));
    csvWrite(member_forces, outdir + "/member_forces.tsv", ascii(9));
    csvWrite(local_displacements, outdir + "/local_displacements.tsv", ascii(9));
    csvWrite(constraint_residual, outdir + "/constraint_residual.tsv", ascii(9));
else
    disp("U =");
    disp(U);
    disp("Lag =");
    disp(Lag);
    disp("R =");
    disp(R);
    disp("local_displacements =");
    disp(local_displacements);
    disp("member_forces =");
    disp(member_forces);
end

exit;
