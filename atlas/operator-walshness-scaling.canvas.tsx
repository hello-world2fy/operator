import {
  Callout,
  Card,
  CardBody,
  CardHeader,
  H1,
  H2,
  LineChart,
  Row,
  Stack,
  Stat,
  Table,
  Text,
} from "cursor/canvas";

const BETAS = ["0", "0.125", "0.25", "0.5", "1", "2", "4", "8", "16"];

export default function OperatorWalshnessScaling() {
  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>Fixed-basis Operator Walshness density</H1>
        <Text>
          Open chains, J = 1, α = 1, computational X basis. The density is W / (2n) and lies in [0, 1]. Infinite temperature is the identity, whose density is log((1 + e²) / 2) / 2 = 0.7169 for every length.
        </Text>
      </Stack>
      <Row gap={16}>
        <Stat value="0.3309" label="TFIM g/J = 1, n = 512, βJ = 16" />
        <Stat value="0.6201" label="Classical ZZ, n = 512, βJ = 16" />
        <Stat value="0.265" label="Mixed-field n = 8 best-found W, not a density" />
      </Row>
      <Card>
        <CardHeader>Transverse-field Ising, g/J = 1</CardHeader>
        <CardBody>
          <H2>Walshness density versus inverse temperature</H2>
          <LineChart
            categories={BETAS}
            beginAtZero
            yMax={1}
            height={280}
            series={[
              { name: "n = 12 exact", data: [0.7169, 0.6123, 0.5085, 0.3803, 0.3188, 0.2987, 0.2864, 0.2795, 0.2783] },
              { name: "n = 32 fermion", data: [0.7169, 0.6122, 0.509, 0.3853, 0.3329, 0.3209, 0.3153, 0.3111, 0.3079] },
              { name: "n = 64 fermion", data: [0.7169, 0.6122, 0.5091, 0.3868, 0.3371, 0.3275, 0.3241, 0.3218, 0.3198] },
              { name: "n = 128 fermion", data: [0.7169, 0.6122, 0.5092, 0.3876, 0.3392, 0.3309, 0.3285, 0.3272, 0.3261] },
              { name: "n = 256 fermion", data: [0.7169, 0.6122, 0.5092, 0.3879, 0.3403, 0.3325, 0.3306, 0.3298, 0.3293] },
              { name: "n = 512 fermion", data: [0.7169, 0.6122, 0.5092, 0.3881, 0.3408, 0.3333, 0.3317, 0.3312, 0.3309] },
            ]}
          />
          <Text>
            Horizontal axis: βJ. Vertical axis: W₁/(2n). Source: exact diagonalization at n = 12 and the open-boundary free-fermion trace for n ≥ 32. At βJ = 16 the density is still rising with n, from 0.3079 at n = 32 to 0.3309 at n = 512.
          </Text>
          <H2>n = 128 MPO against the free-fermion trace</H2>
          <Table
            headers={["βJ", "MPO density", "|ΔW| at α = 1", "χ = 64 vs 128", "discarded weight"]}
            columnAlign={["right", "right", "right", "right", "right"]}
            rows={[
              ["0", "0.7169", "5.1e-13", "0", "0"],
              ["0.125", "0.6122", "8.6e-3", "0", "9.3e-11"],
              ["0.25", "0.5091", "9.8e-3", "0", "2.8e-10"],
              ["0.5", "0.3875", "1.2e-2", "0", "8.8e-10"],
              ["1", "0.3392", "1.7e-2", "3.0e-5", "8.2e-9"],
              ["2", "0.3308", "5.7e-3", "2.2e-2", "1.2e-5"],
            ]}
          />
          <H2>n = 256 MPO, βJ ≤ 1</H2>
          <Table
            headers={["βJ", "MPO density", "|ΔW| at α = 1", "χ = 64 vs 128", "spectrum L1"]}
            columnAlign={["right", "right", "right", "right", "right"]}
            rows={[
              ["0", "0.7169", "2.0e-12", "0", "7.8e-14"],
              ["0.5", "0.3879", "2.4e-2", "0", "1.7e-3"],
              ["1", "0.3402", "3.5e-2", "7.3e-5", "3.0e-3"],
            ]}
          />
          <Text>
            ΔW is the absolute difference of W₁, not the density. Dividing by 2n puts the α = 1 fermion gaps below 7×10⁻⁵ at n = 128 and below 7×10⁻⁵ at n = 256 as well (3.5×10⁻² / 512). Bond dimensions 64 and 128 agree through βJ = 1 at both lengths. At n = 128 and βJ = 2 the α = 1 bond-dimension gap is 2.2×10⁻², while α = 2 moves by 5.8×10⁻². The degree-histogram L1 distance, 1.2×10⁻³ at n = 128 and βJ = 0.5 and 3.0×10⁻³ at n = 256 and βJ = 1, tracks the Trotter step dt = 0.05. The classical Ising chain at n = 256 matches the transfer matrix to 10⁻⁸ at every listed temperature, with bond dimension 2.
          </Text>
        </CardBody>
      </Card>
      <Card>
        <CardHeader>Mixed-field Ising, (g/J, h/J) = (0.8, 0.2)</CardHeader>
        <CardBody>
          <Table
            headers={["βJ", "n = 128 density", "n = 256 density", "χ used", "discarded weight"]}
            columnAlign={["right", "right", "right", "right", "right"]}
            rows={[
              ["0", "0.7169", "0.7169", "1", "0"],
              ["0.25", "0.5496", "0.5496", "10", "4e-10"],
              ["0.5", "0.4622", "0.4627", "22", "2e-9"],
              ["1", "0.4434", "0.4445", "64", "1e-8"],
            ]}
          />
          <Text>
            Fixed X basis, α = 1, open chain, dt = 0.05. Density is W₁/(2n). Bond dimensions 64 and 128 give the same Walshness at every listed point. The bond dimension actually used is unchanged from n = 128 to n = 256, and the density moves by at most 0.0011. There is no fermion reference for this model; the bond-dimension comparison is the convergence check. At βJ = 1 the four mixed-field points on n = 256 have densities 0.4445, 0.3808, 0.4157, and 0.3839 for (g/J, h/J) = (0.8, 0.2), (1.05, 0.2), (1.05, 0.5), and (1.2, 0.5). The same densities at n = 128 are 0.4434, 0.3795, 0.4148, and 0.3828. The largest χ = 64 versus 128 gap in W is 4.7×10⁻⁴, and the discarded weight is about 10⁻⁸. The same check fails for open critical TFIM at n = 16 once βJ ≥ 8: at βJ = 16 the fermion value is 9.244, χ = 64 returns 9.436, and χ = 128 returns 9.917, with discarded weight still near 0.08. Those low-temperature MPO numbers are not usable. At n = 12 the same low-temperature error is mostly truncation: χ = 256 at βJ = 8 brings the α = 1 gap from 0.056 down to 7.5×10⁻⁴, while α = 2 still misses by 8.6×10⁻³.
          </Text>
        </CardBody>
      </Card>
      <Card>
        <CardHeader>Mixed-field (1.05, 0.5), fixed X, exact sizes</CardHeader>
        <CardBody>
          <Table
            headers={["βJ", "n = 12 density", "n = 14 density"]}
            columnAlign={["right", "right", "right"]}
            rows={[
              ["0", "0.7169", "0.7169"],
              ["0.125", "0.6065", "0.6065"],
              ["0.25", "0.5034", "0.5036"],
              ["0.5", "0.4206", "0.4225"],
              ["1", "0.3970", "0.3998"],
              ["2", "0.3929", "0.3960"],
              ["4", "0.3927", "—"],
            ]}
          />
          <Text>
            α = 1, open chain, density W₁/(2n). The n = 12 column is the saved exact diagonalization. The n = 14 column is the Gibbs value printed by the job that is still computing that Hamiltonian; its npz is written only after the eigenstate and microcanonical tables finish. From n = 12 to n = 14 the fixed-X density changes by at most 0.003. At n = 12 it has already flattened by βJ = 4. That plateau, 0.393, is the fixed-X value. The stationary basis search at the same point reaches an upper bound 0.197, density 0.0082, which is still above the floor.
          </Text>
        </CardBody>
      </Card>
      <Card>
        <CardHeader>n = 16 Hutchinson trace versus free fermions</CardHeader>
        <CardBody>
          <Table
            headers={["Probes", "Finished points", "Median |ΔW|", "Inside 2 standard errors"]}
            columnAlign={["right", "right", "right", "right"]}
            rows={[
              ["32", "17", "0.025", "17/17"],
              ["128", "16", "0.0095", "15/16"],
              ["512", "16", "0.011", "13/16"],
            ]}
          />
          <Text>
            Open critical transverse Ising, fixed X basis, files on disk. The sample now reaches βJ = 2. ΔW is the absolute error against the free-fermion trace. The same probes enter the numerator and the denominator, so the estimate stays slightly biased: the median error falls from 32 to 128 probes and then stops falling at 512. A second seed is stored as a held-out repeat. This check does not produce the degree histogram.
          </Text>
          <H2>Larger chains, 32 probes unless noted</H2>
          <Table
            headers={["n", "Probes", "Finished points", "Median |ΔW|", "Inside 2 standard errors"]}
            columnAlign={["right", "right", "right", "right", "right"]}
            rows={[
              ["18", "32", "20", "0.042", "20/20"],
              ["18", "128", "19", "0.015", "18/19"],
              ["20", "32", "19", "0.032", "16/19"],
              ["22", "32", "12", "0.0083", "5/12"],
            ]}
          />
          <Text>
            Still critical transverse Ising, and still an unfinished run. At n = 22 the reported standard error is smaller than the same-probe bias. At infinite temperature and α = 2, 32 probes give 36.409 ± 0.031 against the identity value 36.575. The gap is 0.166. More probes shrink the quoted interval without removing that bias. Mixed-field traces have not been written.
          </Text>
          <H2>α = 1, three-way overlap</H2>
          <Table
            headers={["βJ", "Free fermion", "MPO χ = 128", "128 probes", "512 probes"]}
            columnAlign={["right", "right", "right", "right", "right"]}
            rows={[
              ["0", "22.9405", "22.9405", "22.9406±0.022", "22.9266±0.005"],
              ["0.125", "19.5934", "19.5924", "19.5913±0.008", "19.5870±0.003"],
              ["0.25", "16.2778", "16.2767", "16.2836±0.001", "16.2752±0.004"],
              ["0.5", "12.2340", "12.2327", "12.2376±0.020", "12.2253±0.014"],
              ["2", "9.8427", "9.8402", "9.8379±0.085", "—"],
            ]}
          />
          <Text>
            At βJ ≤ 0.5 the MPO with dt = 0.05 sits within 1.3×10⁻³ of the fermion value. At βJ = 2 the same MPO is 9.8402 against the fermion value 9.8427, a gap of 2.4×10⁻³. The 128-probe estimate is 9.8379±0.085 and agrees; 32 probes give 9.969±0.432. The 512-probe sample at βJ = 2 has not been written. Where both probe counts exist, the 512-probe estimate can land several reported standard errors away while the 128-probe estimate is closer, which is the same-probe bias rather than a failure of the MPO.
          </Text>
        </CardBody>
      </Card>
      <Card>
        <CardHeader>n = 12 critical TFIM, imaginary-time step</CardHeader>
        <CardBody>
          <Table
            headers={["βJ", "Exact W₁", "dt = 0.1", "dt = 0.05", "dt = 0.025", "worst α at dt = 0.025"]}
            columnAlign={["right", "right", "right", "right", "right", "right"]}
            rows={[
              ["0.5", "9.1275", "9.1238", "9.1265", "9.1272", "2.5e-4"],
              ["1", "7.6513", "7.6461", "7.6500", "7.6510", "4.0e-4"],
              ["2", "7.1692", "7.1612", "7.1672", "7.1687", "5.8e-4"],
              ["4", "6.8747", "6.8688", "6.8673", "6.8710", "5.7e-3"],
            ]}
          />
          <Text>
            Bond dimension 128. Through βJ = 2 the error drops by about a factor of four when the step is halved, which is the second-order Trotter scaling, and the discarded weight stays below 10⁻⁸. At βJ = 4 the halved step still misses by 5.7×10⁻³ because the bond dimension is saturated. Raising it to 256 at dt = 0.05 brings the worst error to 2.9×10⁻³. Bond dimension 512, with discarded weight 4×10⁻⁹, leaves that same 2.8×10⁻³, so the remainder is the time step. At βJ = 8, bond dimension 256 still has a discarded weight of 10⁻³ and an α = 2 error of 8.6×10⁻³. Bond dimension 512 drops the discarded weight to 1.5×10⁻⁷ and leaves a worst-case error of 2.7×10⁻³, now on the low side of the exact value, which is the Trotter bias once truncation is gone. At βJ = 16, bond dimension 128 missed α = 1 by 0.20. Bond dimension 256 brings that error to 4.1×10⁻³, while α = 2 still misses by 2.4×10⁻² and the discarded weight is 4.9×10⁻³. Bond dimension 512, after a stable singular-value retry, has discarded weight 1.8×10⁻⁶. Every α then sits within 2.5×10⁻³ of exact diagonalization, the same Trotter residual seen at βJ = 8. The degree-histogram L1 distance falls from 3.5×10⁻² at bond dimension 128 to 8.5×10⁻⁴ at 512.
          </Text>
        </CardBody>
      </Card>
      <Card>
        <CardHeader>Classical ZZ Ising</CardHeader>
        <CardBody>
          <H2>Walshness density versus inverse temperature</H2>
          <LineChart
            categories={BETAS}
            beginAtZero
            yMax={1}
            height={280}
            series={[
              { name: "n = 12 exact", data: [0.7169, 0.7128, 0.702, 0.6718, 0.6333, 0.6211, 0.6201, 0.6201, 0.6201] },
              { name: "n = 32 transfer", data: [0.7169, 0.7126, 0.7011, 0.6696, 0.6309, 0.6206, 0.6201, 0.6201, 0.6201] },
              { name: "n = 128 transfer", data: [0.7169, 0.7125, 0.7008, 0.6686, 0.6298, 0.6204, 0.6201, 0.6201, 0.6201] },
              { name: "n = 512 fixed X", data: [0.7169, 0.7125, 0.7007, 0.6683, 0.6295, 0.6203, 0.6201, 0.6201, 0.6201] },
              { name: "n = 512 staggered ±Z", data: [0.7169, 0.6491, 0.5961, 0.5368, 0.5052, 0.5002, 0.5, 0.5, 0.5] },
            ]}
          />
          <Text>
            Horizontal axis: βJ. Vertical axis: W₁/(2n). The n = 12, 32 and 128 curves, and the n = 512 fixed-X curve, are the computational X basis: exact diagonalization at n = 12 and the two-replica transfer matrix for longer chains. By βJ = 4 that fixed-X density is 0.6201 for every length shown. The staggered curve uses even sites on +Z and odd sites on −Z. It reproduces every n = 12 multi-start value at α = 1 to 10⁻¹⁴. At n = 512 it reaches the floor density 0.5 by βJ = 8, where W = n, so that value is the global minimum. Above the floor the staggered number is exact in that basis and still only an upper bound on the intrinsic minimum.
          </Text>
        </CardBody>
      </Card>
      <Card>
        <CardHeader>βJ = 16, fixed X basis</CardHeader>
        <CardBody>
          <Table
            headers={["n", "TFIM g/J = 1", "Classical ZZ"]}
            columnAlign={["right", "right", "right"]}
            rows={[
              ["12", "0.2783", "0.6201"],
              ["14", "0.2842", "—"],
              ["32", "0.3079", "0.6201"],
              ["64", "0.3198", "0.6201"],
              ["128", "0.3261", "0.6201"],
              ["256", "0.3293", "0.6201"],
              ["512", "0.3309", "0.6201"],
            ]}
          />
          <Text>Entries are W₁/(2n). The n = 14 transverse-field value is exact diagonalization and agrees with the free-fermion result to 1×10⁻¹². Its classical ZZ entry is blank because that Gibbs matrix has not been saved. ZZ at n = 64 and 256 is omitted because n = 32, 128 and 512 already agree to four digits.</Text>
        </CardBody>
      </Card>
      <Card>
        <CardHeader>Exact eigenstates, fixed X basis, TFIM g/J = 1</CardHeader>
        <CardBody>
          <H2>Mean Walshness density across the energy spectrum</H2>
          <LineChart
            categories={["-1.1", "-0.8", "-0.5", "-0.2", "0.2", "0.5", "0.8", "1.1"]}
            beginAtZero
            yMax={1}
            height={280}
            series={[
              { name: "n = 8", data: [0.3075, 0.372, 0.445, 0.5149, 0.5866, 0.6636, 0.7498, 0.8321] },
              { name: "n = 10", data: [0.3149, 0.381, 0.4479, 0.5179, 0.5915, 0.6667, 0.7441, 0.829] },
              { name: "n = 12", data: [0.3226, 0.385, 0.4509, 0.5197, 0.5913, 0.6655, 0.7431, 0.8235] },
              { name: "n = 14", data: [0.3325, 0.3889, 0.4532, 0.5211, 0.5919, 0.6651, 0.7408, 0.8209] },
            ]}
          />
          <Text>
            Horizontal axis: center of an equal-width bin of E/n, edges from −1.25 to 1.25. Vertical axis: mean of W₁/(2n) over the eigenstates in that bin. Open boundary, α = 1, computational X basis. The n = 14 ground state has density 0.284 and the highest eigenstate 0.930; at n = 12 those values are 0.278 and 0.932. Half-chain entanglement and this Walshness density remain nearly uncorrelated (about −0.13 at n = 14 and −0.12 at n = 12).
          </Text>
        </CardBody>
      </Card>
      <Card>
        <CardHeader>Microcanonical window versus matched Gibbs state</CardHeader>
        <CardBody>
          <H2>n = 12 critical TFIM, α = 1</H2>
          <LineChart
            categories={["1", "2", "4", "8", "16", "32"]}
            beginAtZero
            yMax={1}
            height={280}
            referenceLines={[{ value: 0.7169, label: "infinite T" }]}
            series={[
              { name: "window at E/n = -0.52", data: [0.4927, 0.4676, 0.4599, 0.4559, 0.4555, 0.4503] },
              { name: "Gibbs at βJ = 0.29", data: [0.4777, 0.4775, 0.4774, 0.4774, 0.4775, 0.4774] },
              { name: "window at E/n = 0", data: [0.5951, 0.5754, 0.568, 0.5695, 0.5618, 0.5579] },
              { name: "Gibbs at βJ = 0", data: [0.7169, 0.7169, 0.7169, 0.7169, 0.7169, 0.7169] },
            ]}
          />
          <Text>
            Horizontal axis: number of consecutive eigenstates in the window. Vertical axis: W₁/(2n) of the normalized projector. The Gibbs curves use the inverse temperature whose canonical energy matches the window. At E/n = −0.52 the 32-state window, density 0.450, sits close to its Gibbs value 0.477. At the spectrum center the same window stays near 0.558, while the matched Gibbs state is the infinite-temperature identity at 0.717. The n = 14 spectrum shows the same split: a 32-state window at E/n = −0.48 has density 0.458 against Gibbs 0.496, and a window at E/n = 0 stays at 0.558 against 0.717. Rank 1 is one eigenvector from the dense diagonalization; a degenerate partner can differ. Rank R is the projector onto those R states.
          </Text>
        </CardBody>
      </Card>
      <Card>
        <CardHeader>Fixed-basis Walshness and operator entanglement, n = 12</CardHeader>
        <CardBody>
          <H2>Walshness density versus inverse temperature</H2>
          <LineChart
            categories={["0", "0.125", "0.25", "0.5", "1", "2", "4", "8", "16"]}
            beginAtZero
            yMax={1}
            height={240}
            series={[
              { name: "TFIM g/J = 1", data: [0.7169, 0.6123, 0.5085, 0.3803, 0.3188, 0.2987, 0.2864, 0.2795, 0.2783] },
              { name: "mixed (0.8, 0.2)", data: [0.7169, 0.6333, 0.5493, 0.4525, 0.4234, 0.4175, 0.4168, 0.4168, 0.4168] },
              { name: "XXZ Δ = 1", data: [0.7169, 0.6417, 0.5817, 0.5372, 0.5161, 0.5024, 0.5, 0.5, 0.5] },
            ]}
          />
          <H2>Half-chain operator entanglement (bits)</H2>
          <LineChart
            categories={["0", "0.125", "0.25", "0.5", "1", "2", "4", "8", "16"]}
            beginAtZero
            height={240}
            series={[
              { name: "TFIM g/J = 1", data: [0, 0.1093, 0.2776, 0.521, 0.7341, 0.9077, 1.05, 1.1303, 1.1439] },
              { name: "mixed (0.8, 0.2)", data: [0, 0.1099, 0.2711, 0.3088, 0.0543, 0.0225, 0.0223, 0.0223, 0.0223] },
              { name: "XXZ Δ = 1", data: [0, 0.3605, 0.8408, 1.321, 1.5595, 1.5664, 1.5493, 1.549, 1.549] },
            ]}
          />
          <Text>
            Horizontal axis: βJ. Exact diagonalization, open chains, α = 1, computational X basis. Operator entanglement is the half-chain Schmidt entropy of the vectorized Gibbs operator, in bits. For the transverse-field and XXZ chains the two curves move in opposite directions as the system is cooled. For mixed-field Ising at (g/J, h/J) = (0.8, 0.2), operator entanglement falls back to 0.022 bits by βJ = 4, while the fixed-X Walshness density stays at 0.417. That fixed-basis value is not the intrinsic minimum.
          </Text>
        </CardBody>
      </Card>
      <Card>
        <CardHeader>Walsh degree distribution, n = 12, fixed X basis</CardHeader>
        <CardBody>
          <H2>log10 of the degree probability</H2>
          <LineChart
            categories={["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24"]}
            beginAtZero={false}
            yMin={-16}
            yMax={0}
            height={280}
            series={[
              { name: "TFIM g/J = 1, βJ = 0", data: [-3.612,-16,-2.533,-16,-1.793,-16,-1.27,-16,-0.918,-16,-0.714,-16,-0.647,-16,-0.714,-16,-0.918,-16,-1.27,-16,-1.793,-16,-2.533,-16,-3.612] },
              { name: "TFIM g/J = 1, βJ = 1", data: [-1.091,-16,-0.597,-16,-0.489,-16,-0.651,-16,-1.039,-16,-1.638,-16,-2.444,-16,-3.464,-16,-4.713,-16,-6.222,-16,-8.042,-16,-10.274,-16,-13.153] },
              { name: "TFIM g/J = 1, βJ = 16", data: [-0.837,-16,-0.486,-16,-0.512,-16,-0.798,-16,-1.304,-16,-2.013,-16,-2.922,-16,-4.038,-16,-5.381,-16,-6.982,-16,-8.902,-16,-11.274,-16,-14.143] },
              { name: "mixed (0.8, 0.2), βJ = 16", data: [-3.158,-2.3,-1.728,-1.328,-1.053,-0.88,-0.793,-0.785,-0.847,-0.976,-1.169,-1.424,-1.741,-2.12,-2.561,-3.066,-3.639,-4.283,-5.003,-5.808,-6.71,-7.724,-8.878,-10.222,-11.882] },
              { name: "classical ZZ, βJ = 16", data: [-6.924,-16,-4.483,-16,-2.897,-16,-1.795,-16,-1.057,-16,-0.631,-16,-0.492,-16,-0.631,-16,-1.057,-16,-1.795,-16,-2.897,-16,-4.483,-16,-6.924] },
            ]}
          />
          <Text>
            Horizontal axis: Walsh degree k = |r| + |q|, from 0 to 2n. Vertical axis: log10 P(k), floored at 10⁻¹⁶. Exact diagonalization. Transverse-field Ising and classical ZZ put essentially all of their weight on even degrees. The mixed-field chain at βJ = 16 fills the odd degrees as well. Cooling the critical transverse-field chain moves weight from the center of the spectrum toward low degree, but a visible tail remains past k = 12.
          </Text>
        </CardBody>
      </Card>
      <Card>
        <CardHeader>Local Walsh-degree correlation, n = 12, fixed X basis</CardHeader>
        <CardBody>
          <Table
            headers={["Model", "βJ = 1 diagonal", "βJ = 1 neighbor", "βJ = 16 neighbor"]}
            columnAlign={["left", "right", "right", "right"]}
            rows={[
              ["TFIM g/J = 1", "0.323", "0.073", "0.079"],
              ["mixed (0.8, 0.2)", "0.430", "0.039", "0.031"],
              ["classical ZZ", "0.552", "0.000", "0.000"],
              ["XXZ Δ = 1", "0.575", "-0.370", "-0.312"],
            ]}
          />
          <Text>
            Entries are the connected correlation of the local Walsh degree, averaged over bonds for the neighbor column and over sites for the diagonal. Exact diagonalization. At infinite temperature every model has diagonal 1 and neighbor correlation 0. Classical ZZ stays uncorrelated at every temperature in this fixed X basis. XXZ develops a negative nearest-neighbor correlation of about −0.37 by βJ = 1.
          </Text>
        </CardBody>
      </Card>
      <Card>
        <CardHeader>Optimized-basis degree correlation, n = 8, α = 1</CardHeader>
        <CardBody>
          <Table
            headers={["Model", "βJ", "Fixed neighbor", "Optimized neighbor", "Optimized diagonal"]}
            columnAlign={["left", "right", "right", "right", "right"]}
            rows={[
              ["TFIM g/J = 1", "1", "0.075", "0.075", "0.311"],
              ["TFIM g/J = 1", "16", "0.083", "0.083", "0.218"],
              ["mixed (0.8, 0.2)", "1", "0.044", "0.025", "0.240"],
              ["mixed (0.8, 0.2)", "16", "0.035", "0.005", "0.009"],
              ["classical ZZ", "1", "0.000", "−0.964", "1.000"],
              ["classical ZZ", "16", "0.000", "0.143", "1.000"],
              ["XXZ Δ = 1", "1", "−0.373", "−0.373", "0.567"],
              ["XXZ Δ = 1", "16", "−0.321", "−0.321", "0.500"],
            ]}
          />
          <Text>
            Neighbor entries average the connected correlation of d_i = r_i + q_i over bonds. The optimized columns use the saved physical basis, after rotating each local axis onto Z. The moment of that rotated operator reproduces the stored Walshness to 3×10⁻¹² for all 1,848 files at n = 6, 8 and 10. Critical TFIM and XXZ stay on the fixed-X correlations. Mixed-field (0.8, 0.2) at βJ = 16 collapses to diagonal 0.009; that basis is an upper bound, W = 0.265. Classical ZZ at βJ = 16 has W = 8, equal to the α→0 floor, so its correlations belong to the global minimum. At βJ = 1 the same model has neighbor correlation −0.964 in a basis whose value 8.174 is still an upper bound.
          </Text>
        </CardBody>
      </Card>
      <Card>
        <CardHeader>Optimized-basis degree correlation, n = 12, α = 1</CardHeader>
        <CardBody>
          <Table
            headers={["Model", "βJ", "Diagonal", "Neighbor", "Status"]}
            columnAlign={["left", "right", "right", "right", "left"]}
            rows={[
              ["classical ZZ, open", "1", "1.000", "−0.964", "upper bound"],
              ["classical ZZ, open", "16", "1.000", "−1.000", "global minimum"],
              ["classical ZZ, periodic", "16", "1.000", "−1.000", "global minimum"],
              ["mixed (0.8, 0.2)", "1", "0.236", "0.020", "upper bound"],
              ["mixed (0.8, 0.2)", "16", "0.007", "0.004", "upper bound"],
              ["mixed (1.05, 0.5)", "1", "0.013", "0.004", "upper bound"],
              ["mixed (1.05, 0.5)", "16", "0.005", "0.003", "upper bound"],
              ["mixed (1.2, 0.5)", "16", "0.008", "0.004", "upper bound"],
              ["XXZ Δ = 0.5", "16", "0.500", "−0.273", "global minimum"],
              ["XXZ Δ = 1", "16", "0.500", "−0.312", "global minimum"],
              ["TFIM g/J = 1", "16", "0.238", "0.079", "fixed X"],
            ]}
          />
          <Text>
            Connected correlation of the local Walsh degree after rotating each saved axis onto Z. The moment of that operator matches the stored Walshness to at most 8×10⁻¹². Classical ZZ at βJ = 16 has neighbor correlation −1, the same perfect alternation as its ±Z texture. The mixed-field upper bounds are nearly uncorrelated. XXZ meets the floor W = 12 and keeps a negative neighbor correlation. Critical TFIM remains the fixed-X correlation.
          </Text>
        </CardBody>
      </Card>
      <Card>
        <CardHeader>Saved local axes, n = 12, α = 1</CardHeader>
        <CardBody>
          <Table
            headers={["Model", "βJ", "Texture", "Neighbor nᵢ·nᵢ₊₁", "W", "Fixed X"]}
            columnAlign={["left", "right", "left", "right", "right", "right"]}
            rows={[
              ["classical ZZ, open", "16", "staggered ±Z", "−1.000", "12.000", "14.883"],
              ["classical ZZ, open", "1", "staggered ±Z", "−1.000", "12.245", "15.200"],
              ["XXZ Δ = 0.5", "16", "uniform +Z", "+1.000", "12.000", "12.116"],
              ["XXZ Δ = 1", "2", "uniform +Z", "+1.000", "12.058", "12.058"],
              ["XXZ Δ = 1", "16", "uniform +Z", "+1.000", "12.000", "12.000"],
              ["mixed (0.8, 0.2)", "16", "smooth, mostly +Z", "+0.995", "0.317", "10.002"],
              ["mixed (1.05, 0.5)", "16", "smooth, mostly +Z", "+0.997", "0.197", "9.426"],
              ["mixed (1.2, 0.5)", "16", "smooth, mostly +Z", "+0.996", "0.308", "8.605"],
              ["TFIM g/J = 1, open", "16", "uniform +X", "+1.000", "6.678", "6.678"],
              ["TFIM g/J = 1.5", "16", "uniform +X", "+1.000", "3.374", "3.374"],
            ]}
          />
          <Text>
            Each row is the basis stored for that point. Neighbor alignment is the average of nᵢ·nᵢ₊₁. Classical ZZ at βJ = 16 has W = 12, equal to the α→0 floor, and the axis is exactly (+Z, −Z) on alternate sites, with no Y component. XXZ at Δ = 0.5 and βJ = 16 also meets the floor, but with every site on +Z. The mixed-field rows are stationary upper bounds: the bulk axis has n_z ≈ 0.90 and n_x ≈ 0.40, the edges tilt back toward X, and n_y is numerically zero. Critical TFIM, including g/J = 1.5 and the periodic chain, remains on +X. At mixed (0.8, 0.2) and βJ = 1 the saved axis is mostly +X with one site, index 6, flipped to n_z = −0.29; that value 6.125 is an upper bound, and the original 15-restart spread was not retained.
          </Text>
          <H2>mixed (0.8, 0.2), βJ = 1, several α</H2>
          <Table
            headers={["α", "Upper bound", "Fixed X", "Floor W₀"]}
            columnAlign={["right", "right", "right", "right"]}
            rows={[
              ["0.25", "0.753", "7.623", "0.343"],
              ["0.5", "2.355", "8.455", "0.343"],
              ["1", "6.125", "10.162", "0.343"],
              ["2", "9.582", "13.354", "0.343"],
            ]}
          />
          <Text>
            Same chain and temperature. The α = 0.25, 0.5 and 1 values are stationary. α = 2 still has a gradient norm of 9×10⁻⁵, so 9.582 can move slightly. All four sit far above the floor 0.343 and far below fixed X.
          </Text>
          <H2>mixed (0.8, 0.2), βJ = 2</H2>
          <Table
            headers={["α", "Upper bound", "Fixed X", "Floor W₀"]}
            columnAlign={["right", "right", "right", "right"]}
            rows={[
              ["0.25", "0.120", "7.601", "0.088"],
              ["0.5", "0.176", "8.382", "0.088"],
              ["1", "1.844", "10.019", "0.088"],
              ["2", "6.939", "13.157", "0.088"],
            ]}
          />
          <Text>
            Stationary upper bounds on the same n = 12 chain. Smaller α sits closer to the floor: 0.120 at α = 0.25 versus 1.844 at α = 1, and 6.939 at α = 2. Fixed X stays near 8–13. The α = 2 gradient is 3×10⁻⁸ after a further quasi-Newton polish. None of these equals the floor 0.088.
          </Text>
        </CardBody>
      </Card>
      <Card>
        <CardHeader>Operator-network pilot, 80 Adam steps</CardHeader>
        <CardBody>
          <H2>n = 8</H2>
          <Table
            headers={["Setting", "Best relative squared error", "Walsh tail at 10⁻³"]}
            columnAlign={["left", "right", "right"]}
            rows={[
              ["TFIM g/J = 1, βJ = 4, computational basis", "0.282", "8"],
              ["TFIM g/J = 1, βJ = 1, computational basis", "0.394", "10"],
              ["TFIM g/J = 1, βJ = 0.25, computational basis", "0.744", "12"],
              ["mixed (0.8, 0.2), βJ = 4, α = 2 basis", "1.000", "15"],
            ]}
          />
          <H2>n = 10</H2>
          <Table
            headers={["Setting", "Best relative squared error", "Walsh tail at 10⁻³"]}
            columnAlign={["left", "right", "right"]}
            rows={[
              ["TFIM g/J = 1, βJ = 4, computational basis", "0.425", "10"],
              ["TFIM g/J = 1, βJ = 1, computational basis", "0.544", "10"],
              ["TFIM g/J = 1, βJ = 0.25, computational basis", "0.854", "14"],
              ["mixed (0.8, 0.2), βJ = 4, computational basis", "0.934", "13"],
            ]}
          />
          <H2>n = 12, computational and one random basis</H2>
          <Table
            headers={["Setting", "Best relative squared error", "Walsh tail at 10⁻³"]}
            columnAlign={["left", "right", "right"]}
            rows={[
              ["TFIM g/J = 1, βJ = 4, computational basis", "0.564", "12"],
              ["TFIM g/J = 1, βJ = 1, computational basis", "0.671", "12"],
              ["TFIM g/J = 1, βJ = 0.25, computational basis", "0.926", "16"],
              ["mixed (0.8, 0.2), βJ = 4, computational basis", "0.977", "15"],
            ]}
          />
          <Text>
            The error is ||prediction − operator||² / ||operator||² over every matrix element. Predicting zero scores 1. At n = 8, square networks have median 0.88 and tanh networks stay near 0.97; the best error correlates 0.74 with the Walsh tail across 30 pairs. At n = 10 the square median is 0.94, the tanh median is 0.99, and the correlation is 0.62. At n = 12 only the computational basis and one shared random basis were fit, 240 runs, because the optimized directions for these two models are not saved yet. The square median is 0.98 and the tanh median is 1.00. The best fit is again low-temperature critical TFIM in the computational basis, with error 0.564 and tail 12. Optimized bases at n = 10, which are upper bounds on intrinsic Walshness, leave tails near degree 20 and errors near 1, because that rotation makes the matrix sparse and the bit-input network then sees a high-degree function.
          </Text>
        </CardBody>
      </Card>
      <Card>
        <CardHeader>n = 12 classical Ising, α = 1, open chain</CardHeader>
        <CardBody>
          <Table
            headers={["βJ", "Best W found", "Fixed X", "Exact floor W₀"]}
            columnAlign={["right", "right", "right", "right"]}
            rows={[
              ["0", "17.205", "17.205", "12"],
              ["0.5", "13.108", "16.124", "12"],
              ["1", "12.245", "15.200", "12"],
              ["2", "12.006", "14.906", "12"],
              ["4", "12.000002", "14.883", "12"],
              ["8", "12.000000", "14.883", "12"],
              ["16", "12.000000", "14.883", "12"],
            ]}
          />
          <Text>
            The thermal state is symmetric under a global spin flip, so every local Bloch vector of ρ² vanishes and the α→0 value is exactly n = 12 at every temperature. That number is a lower bound for every α. At βJ = 8 and βJ = 16 the search returns 12 to 10⁻¹³, so those values are the global minimum. At βJ = 4 the excess is 2×10⁻⁶. The periodic chain does the same: βJ = 8 and 16 also sit on 12. The 15 restarts still spread by about 11, and the gradient tolerance is met. Fixed-X Walshness stays on the transfer-matrix plateau near 14.883.
          </Text>
        </CardBody>
      </Card>
      <Callout tone="warning" title="Basis search values are upper bounds">
        At n = 8, α = 1, βJ = 16, mixed-field Ising with (g/J, h/J) = (0.8, 0.2), 24 restarts agree to 10⁻¹⁴ and return W = 0.265 against a fixed-X value of 6.468. That minimum is not certified. The same search on critical TFIM at n = 8, βJ = 4 stays on the fixed-X value 4.241. At n = 12 and α = 1, twenty quasi-Newton iterations bring the same mixed-field chain to stationary upper bounds 6.125 at βJ = 1 and 0.318 at βJ = 4 and 16. The α→0 floors are 0.343 and 0.085, so these stationary values still sit above the floor. Fixed-X Walshness is about 10. For (g/J, h/J) = (1.05, 0.5) the stationary upper bounds are 0.663 at βJ = 1, against a fixed-X value of 9.53 and a floor of 0.080. Critical open TFIM at n = 12 and βJ ≥ 1 is stored at the fixed-X value; the search does not beat it. The periodic critical chain at βJ = 16 is 7.974, and g/J = 1.5 at βJ = 16 is 3.374, both equal to fixed X. Mixed-field (1.2, 0.5) has stationary upper bounds 0.721 at βJ = 1 and 0.308 at βJ = 16, against fixed-X values 8.716 and 8.605 and floors 0.113 and 0.090. XXZ at Δ = 0.5 and βJ = 16 returns exactly 12, equal to its α→0 floor, so that value is the global minimum.
      </Callout>
    </Stack>
  );
}
