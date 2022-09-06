import { Group } from "three"
import create from "zustand"
import { useAvatars } from "./avatarStore"
import { immerStore } from "./immerStore"

export enum SelectorMode {
	True,
	Estimated,
	Both,
}

export enum DistanceType {
	Far,
	Near,
}

export enum ContentType {
	Color,
	Normal,
	Inside,
	Outside,
}

export enum SurfaceType {
	Default,
	Project,
}

interface Actions {
	setMode: (mode: SelectorMode) => void
	setDistanceType: (distanceType: DistanceType) => void
	setContentType: (contentType: ContentType) => void
	setSurfaceType: (surfaceType: SurfaceType) => void
}

type Store = {
	mode: SelectorMode
	distanceType: DistanceType
	contentType: ContentType
	surfaceType: SurfaceType

	actions: Actions
}

export const useSelector = create<Store>(
	immerStore((set, get) => ({
		mode: SelectorMode.True,
		distanceType: DistanceType.Far,
		contentType: ContentType.Color,
		surfaceType: SurfaceType.Default,

		actions: {
			setMode: (mode) => {
				set((state) => {
					state.mode = mode
				})
			},

			setDistanceType: (distanceType) => {
				useAvatars.getState().actions.setDistanceType(distanceType == DistanceType.Far ? "far" : "near")
				useAvatars.getState().actions.loadTrueAvatar(() => {
					set((state) => {
						state.distanceType = distanceType
					})
				})
			},
			setContentType: (contentType) => {
				set((state) => {
					state.contentType = contentType
				})
			},
			setSurfaceType: (surfaceType) => {
				set((state) => {
					state.surfaceType = surfaceType
				})
			},
		},
	}))
)
