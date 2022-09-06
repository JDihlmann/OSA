import { Avatar, useAvatars } from "@/stores/avatarStore"
import { ContentType, DistanceType, SelectorMode, SurfaceType, useSelector } from "@/stores/selectorStore"
import { Text } from "@react-three/drei"
import { FunctionComponent, useEffect, useState } from "react"
import PointCloud from "../pointCloud/pointCloud"

interface AvatarManagerProps {}

const AvatarManager: FunctionComponent<AvatarManagerProps> = () => {
	const mode = useSelector((state) => state.mode)
	const distanceType = useSelector((state) => state.distanceType)
	const surfaceType = useSelector((state) => state.surfaceType)
	const contentType = useSelector((state) => state.contentType)

	const trueAvatar = useAvatars((state) => state.trueAvatar)
	const estimatedAvatar = useAvatars((state) => state.estimatedAvatar)
	const reductionPercentage = useAvatars((state) => state.reductionPercentage)

	const [trueAvatarPoints, setTrueAvatarPoints] = useState<number[][] | undefined>()
	const [trueAvatarColors, setTrueAvatarColors] = useState<number[][] | undefined>()

	const [estimatedAvatarPoints, setEstimatedAvatarPoints] = useState<number[][] | undefined>()
	const [estimatedAvatarColors, setEstimatedAvatarColors] = useState<number[][] | undefined>()

	useEffect(() => {
		if (trueAvatar && (mode === SelectorMode.True || mode === SelectorMode.Both)) {
			const colors = adaptAvatarToContentType(trueAvatar, contentType, reductionPercentage)
			setTrueAvatarColors(colors)
		}

		if (estimatedAvatar && (mode === SelectorMode.Estimated || mode === SelectorMode.Both)) {
			const colors = adaptAvatarToContentType(estimatedAvatar, contentType, reductionPercentage)
			setEstimatedAvatarColors(colors)
		}
	}, [contentType, trueAvatar, reductionPercentage, estimatedAvatar, mode])

	useEffect(() => {
		if (trueAvatar && (mode === SelectorMode.True || mode === SelectorMode.Both)) {
			const points = adaptAvatarToSurfaceType(trueAvatar, surfaceType, contentType, reductionPercentage)
			setTrueAvatarPoints(points)
		}

		if (estimatedAvatar && (mode === SelectorMode.Estimated || mode === SelectorMode.Both)) {
			const points = adaptAvatarToSurfaceType(estimatedAvatar, surfaceType, contentType, reductionPercentage)
			setEstimatedAvatarPoints(points)
		}
	}, [contentType, surfaceType, trueAvatar, reductionPercentage, estimatedAvatar, mode])

	return (
		<>
			<group position={mode == SelectorMode.True ? [0, 0, 0] : [-1.5, 0, 0]}>
				{trueAvatar
					? (mode == SelectorMode.True || mode == SelectorMode.Both) &&
					  trueAvatarPoints &&
					  trueAvatarColors && <PointCloud points={trueAvatarPoints} colors={trueAvatarColors} />
					: mode != SelectorMode.Estimated && <Text color={"black"}> Loading</Text>}
			</group>

			<group position={mode == SelectorMode.Estimated ? [0, 0, 0] : [1.5, 0, 0]}>
				{estimatedAvatar
					? (mode == SelectorMode.Estimated || mode == SelectorMode.Both) &&
					  estimatedAvatarPoints &&
					  estimatedAvatarColors && <PointCloud points={estimatedAvatarPoints} colors={estimatedAvatarColors} />
					: mode != SelectorMode.True && <Text color={"black"}> Loading </Text>}
			</group>
		</>
	)
}

export default AvatarManager

const adaptAvatarToContentType = (
	avatar: Avatar,
	contentType: ContentType,
	reductionPercentage: number
): number[][] => {
	let colors: number[][] = []
	switch (contentType) {
		case ContentType.Color:
			colors = sliceToPercentage(avatar.colors, reductionPercentage)
			break
		case ContentType.Normal:
			const normals = sliceToPercentage(avatar.normals, reductionPercentage)
			colors = normalToColor(normals)
			break
		case ContentType.Inside:
			let distances1 = sliceToPercentage(avatar.distances, reductionPercentage)
			distances1 = filterArrForDistance(distances1, distances1, true)
			colors = new Array(distances1.length).fill([0, 0, 1])
			break
		case ContentType.Outside:
			let distances2 = sliceToPercentage(avatar.distances, reductionPercentage)
			distances2 = filterArrForDistance(distances2, distances2, false)
			colors = new Array(distances2.length).fill([1, 0, 0])
	}
	return colors
}

const adaptAvatarToSurfaceType = (
	avatar: Avatar,
	surfaceType: SurfaceType,
	contentType: ContentType,
	reductionPercentage: number
): number[][] => {
	let points: number[][] = []
	switch (surfaceType) {
		case SurfaceType.Default:
			points = sliceToPercentage(avatar.points, reductionPercentage)
			break
		case SurfaceType.Project:
			points = sliceToPercentage(avatar.points, reductionPercentage)
			const normals = sliceToPercentage(avatar.normals, reductionPercentage)
			const distances = sliceToPercentage(avatar.distances, reductionPercentage)
			points = projectToSurface(points, normals, distances)

			break
	}

	if (contentType === ContentType.Inside || contentType === ContentType.Outside) {
		const distances = sliceToPercentage(avatar.distances, reductionPercentage)
		points = filterArrForDistance(points, distances, contentType === ContentType.Inside)
	}
	return points
}

const filterArrForDistance = <T,>(arr: T[], distances: number[], inside: boolean): T[] => {
	return arr.filter((_v, i) => {
		const distance = distances[i] as number
		if (inside) {
			return distance < 0
		} else {
			return distance > 0
		}
	})
}

const sliceToPercentage = <T,>(arr: T[], percentage: number): T[] => {
	return arr.slice(0, Math.floor(arr.length * percentage))
}

const projectToSurface = (points: number[][], normals: number[][], distances: number[]): number[][] => {
	return points.map((point, i) => {
		const distance = distances[i]
		const normal = normals[i] as number[]
		return [point[0] - normal[0] * distance, point[1] - normal[1] * distance, point[2] - normal[2] * distance]
	})
}

const normalToColor = (normals: number[][]): number[][] => {
	return normals.map((normal) => [0.5 + normal[0] / 2, 0.5 + normal[1] / 2, 0.5 + normal[2] / 2])
}
