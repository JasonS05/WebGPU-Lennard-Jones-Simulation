struct Particle {
	position: vec2f,
	velocity: vec2f,
	acceleration: vec2f,
	potential: f32,
	radius: f32,
	coordinationNumber: f32
}

fn equal(a: vec2f, b: vec2f) -> bool {
	return a.x == b.x && a.y == b.y;
}

struct miscData {
	maximumTimeStep: f32,
	inverseTimeStep: atomic<u32>,
	nextMaximumTimeStep: f32,
	time: f32,
	oldTimeStep: f32
}

// values with [[...]] are supplied by JS using textual substitution
const numParticles: u32 = [[numParticles]];
const maxParticleRadius: f32 = [[maxParticleRadius]];
const minParticleRadius: f32 = [[minParticleRadius]];
const potentialCutoff: f32 = [[potentialCutoff]];
const timeStepCaution: f32 = [[timeStepCaution]];
const gravity: f32 = [[gravity]];
const gridCellsPerDimension: i32 = [[gridCellsPerDimension]];
const gridCellSize: f32 = [[gridCellSize]];
const gridCellCapacity: i32 = [[gridCellCapacity]];
const numGridCells: i32 = [[numGridCells]];

const pi = 3.1415926;
const particleSizeAdjustment = 1 / pow(2.0, 1.0/6.0) * 2; // explicit .0 to make Firefox Nightly happy

@group(0) @binding(0) var<storage, read> input_data: array<Particle>;
@group(0) @binding(1) var<storage, read> input_grid: array<Particle>;
@group(0) @binding(2) var<storage, read> input_gridCounters: array<i32>;
@group(0) @binding(3) var<storage, read_write> input_misc: miscData; // read_write for atomic access, no writing is actually done

@group(0) @binding(4) var<storage, read_write> output_data: array<Particle>;
@group(0) @binding(5) var<storage, read_write> output_grid: array<Particle>;
@group(0) @binding(6) var<storage, read_write> output_gridCounters: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> output_misc: miscData;

@compute @workgroup_size(64) fn computeShader(@builtin(global_invocation_id) id: vec3<u32>) {
	let index = id.x;

	// in case the number of particles is not a multiple of the workgroup size
	if (index >= numParticles) {
		return;
	}

	let suggestedTimeStep = input_misc.maximumTimeStep / f32(atomicLoad(&input_misc.inverseTimeStep));
	let timeStep = min(input_misc.oldTimeStep * 0.999 + suggestedTimeStep * 0.001, suggestedTimeStep * 2);

	var particle = input_data[index];
	let mass = 1000 * particle.radius * particle.radius;
	let result = calculateParticleData(particle, timeStep, mass);
	var acceleration = result.force / mass + vec2f(0, -gravity);

	//if (input_misc.time < 10.0) {
	//	acceleration -= particle.velocity * min(1000.0, exp(input_misc.time / 100) / 100); // explicit .0 to make Firefox Nightly happy
	//}

	//if (input_misc.time < 0.00) {
		acceleration -= particle.velocity * 0.1;
	//}

	particle.velocity += (acceleration + particle.acceleration) / 2 * timeStep;
	particle.position += particle.velocity * timeStep + acceleration * timeStep * timeStep / 2;
	particle.acceleration = acceleration;
	particle.potential = result.potential;
	particle.coordinationNumber = result.coordinationNumber;

	particle.position = modulo(particle.position + 1 - particle.radius, 4 - particle.radius * 4);

	if (particle.position.x > 2 - particle.radius * 2) {
		particle.position.x = 4 - particle.radius * 4 - particle.position.x;
		particle.velocity.x *= -1.0; // explicit .0 to make Firefox Nightly happy
	}

	if (particle.position.y > 2 - particle.radius * 2) {
		particle.position.y = 4 - particle.radius * 4 - particle.position.y;
		particle.velocity.y *= -1.0; // explicit .0 to make Firefox Nightly happy
	}

	particle.position -= 1 - particle.radius;

	output_data[index] = particle;
	placeParticleInGrid(particle);

	let minimumInverseTimeStep = u32(1 + floor(timeStepCaution * input_misc.nextMaximumTimeStep * length(particle.velocity) / result.nearestNeighborDist * 2));

	output_misc.maximumTimeStep = input_misc.nextMaximumTimeStep;
	atomicMax(&output_misc.inverseTimeStep, minimumInverseTimeStep);
	output_misc.nextMaximumTimeStep = timeStep * 65536;
	output_misc.time = input_misc.time + timeStep;
	output_misc.oldTimeStep = timeStep;
}

fn modulo(dividend: vec2f, divisor: f32) -> vec2f {
	return dividend - divisor * floor(dividend / divisor);
}

fn placeParticleInGrid(particle: Particle) {
	let gridCellPosition = vec2i(floor((particle.position + 1) / 2 / gridCellSize));
	let gridCellNumberCandidate = gridCellPosition.x + gridCellPosition.y * i32(gridCellsPerDimension);

	if (gridCellNumberCandidate >= 0 && gridCellNumberCandidate < i32(numGridCells)) {
		let gridCellNumber = gridCellNumberCandidate;
		let gridCellOffset = gridCellNumber * gridCellCapacity;
		let gridCellOccupancy = atomicAdd(&output_gridCounters[gridCellNumber], 1);

		if (gridCellOccupancy < gridCellCapacity) {
			output_grid[gridCellOffset + gridCellOccupancy] = particle;
		} else {
			let overflowOffset = numGridCells * gridCellCapacity;
			let overflowOccupancy = atomicAdd(&output_gridCounters[numGridCells], 1);
			output_grid[overflowOffset + overflowOccupancy] = particle;
		}
	}
}

struct particleData {
	force: vec2f,
	potential: f32,
	nearestNeighborDist: f32,
	coordinationNumber: f32
}

fn calculateParticleData(particle: Particle, timeStep: f32, mass: f32) -> particleData {
	let position = particle.position;
	var force = vec2f(0, 0);
	var potential: f32 = 0;
	var nearestNeighborDist = maxParticleRadius * maxParticleRadius * potentialCutoff * potentialCutoff * 4;
	var coordinationNumber: f32 = 0;

	let gridCellPosition = vec2i(floor((particle.position + 1) / 2 / gridCellSize));

	for (var y = gridCellPosition.y - 1; y < gridCellPosition.y + 2; y++) {
		for (var x = gridCellPosition.x - 1; x < gridCellPosition.x + 2; x++) {
			if (x >= 0 && y >= 0 && x < i32(gridCellsPerDimension) && y < i32(gridCellsPerDimension)) {
				let gridCellNumber = x + y * gridCellsPerDimension;
				let gridCellOffset = gridCellNumber * gridCellCapacity;
				let gridCellOccupancy = min(input_gridCounters[gridCellNumber], gridCellCapacity);

				for (var i = gridCellOffset; i < gridCellOffset + gridCellOccupancy; i++) {
					let otherParticle = input_grid[i];

					if (!equal(otherParticle.position, particle.position)) {
						let result = calculateParticleData_helper(particle, otherParticle);

						force += result.force;
						potential += result.potential;
						nearestNeighborDist = min(nearestNeighborDist, result.nearestNeighborDist);
						coordinationNumber += result.coordinationNumber;
					}
				}
			}
		}
	}

	let gridCellOverflowOffset = numGridCells * gridCellCapacity;
	let gridCellOverflowOccupancy = input_gridCounters[numGridCells];

	for (var i = gridCellOverflowOffset; i < gridCellOverflowOffset + gridCellOverflowOccupancy; i++) {
		let otherParticle = input_grid[i];

		if (!equal(otherParticle.position, particle.position)) {
			let result = calculateParticleData_helper(particle, otherParticle);

			force += result.force;
			potential += result.potential;
			nearestNeighborDist = min(nearestNeighborDist, result.nearestNeighborDist);
			coordinationNumber += result.coordinationNumber;
		}
	}

	var result: particleData;
	result.force = force;
	result.potential = potential;
	result.nearestNeighborDist = sqrt(nearestNeighborDist);
	result.coordinationNumber = coordinationNumber;
	return result;
}

fn calculateParticleData_helper(particle1: Particle, particle2: Particle) -> particleData {
	var different: f32 = 0;

	if (particle1.radius != particle2.radius) {
		different = 1.0; // explicit .0 to make Firefox Nightly happy
	}

	let relativePosition = particle2.position - particle1.position;
	let distance2 = dot(relativePosition, relativePosition);
	var summedRadii = (particle1.radius + particle2.radius);

	if (different == 1) {
		summedRadii = minParticleRadius * 2 * 1.618;
	}

	let particleSize = summedRadii * particleSizeAdjustment * 0.5;

	if (distance2 < summedRadii * summedRadii * potentialCutoff * potentialCutoff) {
		let cutoffDistanceRadii = potentialCutoff * particleSizeAdjustment;
		let potentialAtCutoff = pow(cutoffDistanceRadii, -12.0) - pow(cutoffDistanceRadii, -6.0); // explicit .0 to make Firefox Nightly happy

		let iparticleSize = 1 / particleSize;
		let iparticleSize2 = iparticleSize * iparticleSize;
		let radii2 = distance2 * iparticleSize2;
		let iradii2 = 1 / radii2;
		let iradii4 = iradii2 * iradii2;
		let iradii6 = iradii4 * iradii2;
		let iradii12 = iradii6 * iradii6;

		var result: particleData;
		result.force = (iradii6 * 6 - iradii12 * 12) * iradii2 * iparticleSize2 * relativePosition;
		result.potential = iradii12 - iradii6 - potentialAtCutoff;
		result.nearestNeighborDist = distance2;
		result.coordinationNumber = 0.0;

		if (distance2 < summedRadii * summedRadii * 2.1) {
			result.coordinationNumber = 1.0;
		}

		if (different == 0) {
			result.force /= 4.0; // explicit .0 to make Firefox Nightly happy
			result.potential /= 4.0; // explicit .0 to make Firefox Nightly happy
		}

		return result;
	} else {
		return particleData(vec2f(0, 0), 0, maxParticleRadius * maxParticleRadius * potentialCutoff * potentialCutoff * 4, 0);
	}
}

struct VertexShaderOutput {
	@builtin(position) position: vec4f,
	@location(0) color: vec4f
}

@vertex fn vertexShader(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexShaderOutput {
	let particle = input_data[instanceIndex];

	let radius = particle.radius * 0.9;

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
	vsOutput.position = vec4f(particle.position + vertexPosition, 0, 1);

	let red = vec4f(1, 0.5, 0, 1);
	let blue = vec4f(0, 0.5, 1, 1);
	let white = vec4f(1, 1, 1, 1);

	let potentialEnergy = particle.potential / 2;
	let velocity = particle.velocity;
	let kineticEnergy = 0.5 * dot(velocity, velocity);
	let coordinationNumber = particle.coordinationNumber;
	var color = 8 * (potentialEnergy + kineticEnergy);

	/*if (color < 0) {
		vsOutput.color = mix(white, blue, -color);
	} else {
		vsOutput.color = mix(white, red, color);
	}*/

	/*if (radius == maxParticleRadius) {
		vsOutput.color = vec4f(1, 1, 0, 1);
	} else {
		vsOutput.color = vec4f(0, 0, 1, 1);
	}*/

	if (particle.coordinationNumber < 6) {
		vsOutput.color = vec4f(0.5, 0.5, 1, 1);
	} else if (particle.coordinationNumber == 6) {
		vsOutput.color = vec4f(0.5, 1, 0.5, 1);
	} else {
		vsOutput.color = vec4f(1, 0.5, 0.5, 1);
	}

	//vsOutput.color = vec4f(1, 1, 1, 1);

	return vsOutput;
}

@fragment fn fragmentShader(fsInput: VertexShaderOutput) -> @location(0) vec4f {
	return fsInput.color;
}

