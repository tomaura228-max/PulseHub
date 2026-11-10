-- Spins the quadcopter propellers. Put this Script inside the imported Quadcopter model.
-- Propellers are attached to Body_Shell with Motor6D, so it works whether the model is anchored or flying.
local RunService = game:GetService("RunService")

local model = script.Parent
local body = model:WaitForChild("Body_Shell")

local RPS = 12 -- revolutions per second
-- +1 = counter-clockwise seen from above (orange props), -1 = clockwise (black props)
local SPIN = { FL = 1, FR = -1, RL = -1, RR = 1 }

local rotors = {}
for name, dir in pairs(SPIN) do
	local prop = model:FindFirstChild("Propeller_" .. name, true)
	if prop then
		prop.Anchored = false
		prop.CanCollide = false
		prop.Massless = true
		local joint = Instance.new("Motor6D")
		joint.Name = "Rotor_" .. name
		joint.Part0 = body
		joint.Part1 = prop
		joint.C0 = body.CFrame:ToObjectSpace(prop.CFrame)
		joint.Parent = body
		table.insert(rotors, { joint = joint, c0 = joint.C0, dir = dir })
	end
end

local angle = 0
RunService.Heartbeat:Connect(function(dt)
	angle = (angle + dt * RPS * 2 * math.pi) % (2 * math.pi)
	for _, r in ipairs(rotors) do
		r.joint.C0 = r.c0 * CFrame.Angles(0, angle * r.dir, 0)
	end
end)
