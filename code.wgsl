struct Particle {
	position: vec2f,
	velocity: vec2f,
	oldForce: vec2f,
	potential: f32
}

fn equal(p1: Particle, p2: Particle) -> bool {
	return
		p1.position.x == p2.position.x &&
		p1.position.y == p2.position.y &&
		p1.velocity.x == p2.velocity.x &&
		p1.velocity.y == p2.velocity.y &&
		p1.oldForce.x == p2.oldForce.x &&
		p1.oldForce.y == p2.oldForce.y &&
		p1.potential == p2.potential;
}

struct miscData {
	maximumTimeStep: f32,
	inverseTimeStep: atomic<u32>
}

// values with [[...]] are supplied by JS using textual substitution
const numParticles: u32 = [[numParticles]];
const particleRadius: f32 = [[particleRadius]];
const potentialCutoff: f32 = [[potentialCutoff]];
const timeStepCaution: f32 = [[timeStepCaution]];
const gravity: f32 = [[gravity]];
const gridCellsPerDimension: u32 = [[gridCellsPerDimension]];
const gridCellSize: f32 = [[gridCellSize]];
const gridCellCapacity: u32 = [[gridCellCapacity]];
const numGridCells: u32 = [[numGridCells]];

const pi = 3.1415926;
const particleSize = particleRadius / pow(2, 1/6) * 2; // an adjusted version of particleRadius used in calculations

@group(0) @binding(0) var<storage, read> input_data: array<Particle>;
@group(0) @binding(1) var<storage, read> input_grid: array<Particle>;
@group(0) @binding(2) var<storage, read> input_gridCounters: array<u32>;
@group(0) @binding(3) var<storage, read_write> input_misc: miscData; // read_write for atomic access, no writing is actually done

@group(0) @binding(4) var<storage, read_write> output_data: array<Particle>;
@group(0) @binding(5) var<storage, read_write> output_grid: array<Particle>;
@group(0) @binding(6) var<storage, read_write> output_gridCounters: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> output_misc: miscData;

@compute @workgroup_size(64) fn computeShader(@builtin(global_invocation_id) id: vec3<u32>) {
	let index = id.x;

	// in case the number of particles is not a multiple of the workgroup size
	if (index >= numParticles) {
		return;
	}

	let timeStep = input_misc.maximumTimeStep / f32(atomicLoad(&input_misc.inverseTimeStep));

	var particle = input_data[index];
	let result = calculateForceAndPotential(particle, timeStep);

	particle.velocity += (result.force + particle.oldForce) / 2 * timeStep;
	particle.position += particle.velocity * timeStep + result.force * timeStep * timeStep / 2;
	particle.oldForce = result.force;
	particle.potential = result.potential;

	output_data[index] = particle;
	placeParticleInGrid(particle);

	let minimumInverseTimeStep = u32(ceil(timeStepCaution * input_misc.maximumTimeStep * length(particle.velocity) / particleRadius));

	output_misc.maximumTimeStep = input_misc.maximumTimeStep;
	atomicMax(&output_misc.inverseTimeStep, minimumInverseTimeStep);
}

fn placeParticleInGrid(particle: Particle) {
	let gridCellPosition = vec2i(floor((particle.position + 1) / 2 / gridCellSize));
	let gridCellNumberCandidate = gridCellPosition.x + gridCellPosition.y * i32(gridCellsPerDimension);

	if (gridCellNumberCandidate >= 0 && gridCellNumberCandidate < i32(numGridCells)) {
		let gridCellNumber = u32(gridCellNumberCandidate);
		let gridCellOffset = gridCellNumber * gridCellCapacity;
		let gridCellOccupancy = atomicAdd(&output_gridCounters[gridCellNumber], 1);

		if (gridCellOccupancy < gridCellCapacity) {
			output_grid[gridCellOffset + gridCellOccupancy] = particle;
		} else {
			let overflowOffset = numGridCells * gridCellCapacity;
			let overflowOccupancy = atomicAdd(&output_gridCounters[numGridCells], 1);
			output_grid[overflowOffset + overflowOccupancy] = particle;
		}
	} else {
		let n = particleRadius * 2 * potentialCutoff + 1;
		let x = particle.position.x;
		let y = particle.position.y;

		if (x > -n && x < n && y > -n && y < n) {
			let overflowOffset = numGridCells * gridCellCapacity;
			let overflowOccupancy = atomicAdd(&output_gridCounters[numGridCells], 1);
			output_grid[overflowOffset + overflowOccupancy] = particle;
		}
	}
}

struct forceAndPotential {
	force: vec2f,
	potential: f32
}

fn calculateForceAndPotential(particle: Particle, timeStep: f32) -> forceAndPotential {
	let position = particle.position;
	var LJforce = vec2f(0, 0);
	var potential: f32 = 0;

	let gridCellPosition = vec2i(floor((particle.position + 1) / 2 / gridCellSize));

	for (var y = gridCellPosition.y - 1; y < gridCellPosition.y + 2; y++) {
		for (var x = gridCellPosition.x - 1; x < gridCellPosition.x + 2; x++) {
			if (x >= 0 && y >= 0 && x < i32(gridCellsPerDimension) && y < i32(gridCellsPerDimension)) {
				let gridCellNumber = u32(x) + u32(y) * gridCellsPerDimension;
				let gridCellOffset = gridCellNumber * gridCellCapacity;
				let gridCellOccupancy = min(input_gridCounters[gridCellNumber], gridCellCapacity);

				for (var i: u32 = gridCellOffset; i < gridCellOffset + gridCellOccupancy; i++) {
					let otherParticle = input_grid[i];

					if (!equal(otherParticle,particle)) {
						let result = calculateForceAndPotential_helper(particle, otherParticle);

						LJforce += result.force;
						potential += result.potential;
					}
				}
			}
		}
	}

	let gridCellOverflowOffset = numGridCells * gridCellCapacity;
	let gridCellOverflowOccupancy = input_gridCounters[numGridCells];

	for (var i: u32 = gridCellOverflowOffset; i < gridCellOverflowOffset + gridCellOverflowOccupancy; i++) {
		let otherParticle = input_grid[i];

		if (!equal(otherParticle, particle)) {
			let result = calculateForceAndPotential_helper(particle, otherParticle);

			LJforce += result.force;
			potential += result.potential;
		}
	}

	let gravityForce = vec2f(0, -gravity);
	let gravityPotential = gravity * position.y * 2;

	var wallForce = vec2f(0, 0);

	if (position.x > 1) {
		wallForce += vec2f(-position.x + 1, 0);
	}

	if (position.y > 1) {
		wallForce += vec2f(0, -position.y + 1);
	}

	if (position.x < -1) {
		wallForce += vec2f(-position.x - 1, 0);
	}

	if (position.y < -1) {
		wallForce += vec2f(0, -position.y - 1);
	}

	wallForce /= timeStep * timeStep;

	var result: forceAndPotential;
	result.force = LJforce + wallForce + gravityForce;
	result.potential = potential + 0 * gravityPotential;
	return result;
}

fn calculateForceAndPotential_helper(particle1: Particle, particle2: Particle) -> forceAndPotential {
	let relativePosition = particle2.position - particle1.position;
	let distance2 = dot(relativePosition, relativePosition);

	if (distance2 < particleRadius * particleRadius * potentialCutoff * potentialCutoff * 4) {
		let cutoffDistanceRadii = particleRadius * 2 * potentialCutoff / particleSize;
		let potentialAtCutoff = pow(cutoffDistanceRadii, -12) - pow(cutoffDistanceRadii, -6);

		let iparticleSize = 1 / particleSize;
		let iparticleSize2 = iparticleSize * iparticleSize;
		let radii2 = distance2 * iparticleSize2;
		let iradii2 = 1 / radii2;
		let iradii4 = iradii2 * iradii2;
		let iradii6 = iradii4 * iradii2;
		let iradii12 = iradii6 * iradii6;

		var result: forceAndPotential;
		result.force = (iradii6 * 6 - iradii12 * 12) * iradii2 * iparticleSize2 * relativePosition;
		result.potential = iradii12 - iradii6 - potentialAtCutoff;
		return result;
	} else {
		return forceAndPotential(vec2f(0, 0), 0);
	}
}

struct VertexShaderOutput {
	@builtin(position) position: vec4f,
	@location(0) color: vec4f
}

@vertex fn vertexShader(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexShaderOutput {
	let radius = particleRadius;

	var vertexPosition: vec2f;

	// Some black magic to contruct ever more accurate circles with increasing vertex indices. It
	// works by making a central triangle, tacking three onto it to make a hexagon, tacking six
	// more on to make a dodecagon, tacking twelve more on to make a 24-gon, and so on.
	if (vertexIndex < 3) {
		vertexPosition = radius * vec2f(sin(2 * f32(vertexIndex)/3 * pi), cos(2 * f32(vertexIndex)/3 * pi));
	} else {
		var vertexAngleSizeInverse = exp2(floor(log2(floor((floor(f32(vertexIndex) / 3) - 1) / 3 + 1))));
		var v = 6 + f32(vertexIndex) - vertexAngleSizeInverse * 9;
		var vertexAngle = (v - floor(v/3)) / vertexAngleSizeInverse / 6 * 2 * pi;
		vertexPosition = radius * vec2f(sin(vertexAngle), cos(vertexAngle));
	}

	var vsOutput: VertexShaderOutput;
	vsOutput.position = vec4f(input_data[instanceIndex].position + vertexPosition, 0, 1);

	let red = vec4f(1, 0.5, 0, 1);
	let blue = vec4f(0, 0.5, 1, 1);
	let white = vec4f(1, 1, 1, 1);

	let potentialEnergy = input_data[instanceIndex].potential / 2;
	let velocity = input_data[instanceIndex].velocity;
	let kineticEnergy = 0.5 * dot(velocity, velocity);
	var color = 5 * potentialEnergy + 0 * kineticEnergy + 2.5;

	if (color < 0) {
		vsOutput.color = mix(white, blue, -color);
	} else {
		vsOutput.color = mix(white, red, color);
	}

	return vsOutput;
}

@fragment fn fragmentShader(fsInput: VertexShaderOutput) -> @location(0) vec4f {
	return fsInput.color;
}

