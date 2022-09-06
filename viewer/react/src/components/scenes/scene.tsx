import { OrbitControls, Stats } from "@react-three/drei"
import { Canvas } from "@react-three/fiber"

import { FunctionComponent } from "react"
import { NoToneMapping, sRGBEncoding } from "three"
import AvatarManager from "../avatarManager/avatarManager"

interface SceneProps {}

const Scene: FunctionComponent<SceneProps> = ({}) => {
	return (
		<Canvas
			shadows
			dpr={[1, 2]}
			gl={{ antialias: true }}
			onCreated={({ gl }) => {
				gl.toneMapping = NoToneMapping
				gl.outputEncoding = sRGBEncoding
			}}
			camera={{ position: [0, 0, 4] }}
		>
			{/* <color attach="background" args={["#202025"]} /> */}
			<Stats />
			<OrbitControls />

			{/* <mesh>
				<sphereGeometry attach="geometry" args={[20, 100, 100]} />
				<meshBasicMaterial attach="material" color="white" side={1} />
			</mesh> */}
			{/* <group position={[0, -1, 0]}>
				<gridHelper args={[50, 100]} />
			</group> */}

			<AvatarManager />
		</Canvas>
	)
}

export default Scene
