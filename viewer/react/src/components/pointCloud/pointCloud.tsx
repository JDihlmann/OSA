import { FunctionComponent, MutableRefObject, useEffect, useMemo, useRef } from "react"
import { DynamicDrawUsage, Points } from "three"
import "./dotMaterial"

interface PointCloudProps {
	points: number[][]
	colors: number[][]
}

const PointCloud: FunctionComponent<PointCloudProps> = ({ points, colors }) => {
	const bufferPoints = new Float32Array(points.flatMap((v) => v))
	const bufferColors = new Float32Array(colors.flatMap((v) => v))

	const pointsRef: MutableRefObject<Points | null> | undefined = useRef(null)

	useEffect(() => {
		if (pointsRef?.current) {
			pointsRef.current.geometry.attributes.position.needsUpdate = true
			pointsRef.current.geometry.attributes.color.needsUpdate = true
		}
	}, [bufferPoints, bufferColors, pointsRef])

	return (
		<points ref={pointsRef}>
			<bufferGeometry>
				<bufferAttribute
					usage={DynamicDrawUsage}
					attach="attributes-position"
					count={bufferPoints.length / 3}
					array={bufferPoints}
					itemSize={3}
					needsUpdate={true}
				/>
				<bufferAttribute
					usage={DynamicDrawUsage}
					attach="attributes-color"
					count={bufferColors.length / 3}
					array={bufferColors}
					itemSize={3}
					needsUpdate={true}
				/>
			</bufferGeometry>
			<dotMaterial vertexColors />
		</points>
	)
}

export default PointCloud
