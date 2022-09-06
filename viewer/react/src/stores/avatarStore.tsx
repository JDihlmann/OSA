import create from "zustand"
import { immerStore } from "./immerStore"

export interface Avatar {
	points: number[][]
	colors: number[][]
	normals: number[][]
	distances: number[]
}

interface Actions {
	loadTrueAvatar: (callback: (() => void) | undefined | null) => void
	loadEstimatedAvatar: (callback: (() => void) | undefined | null) => void
	setDistanceType: (distanceType: string) => void
	setReductionPercentage: (reductionPercentage: number) => void
	setAvatarName: (avatarName: string) => void
	setEnvironmentName: (environmentName: string) => void
	getNames: () => void
}

type Store = {
	url: string
	distanceType: string

	avatarName: string | undefined
	environmentName: string | undefined
	avatarNames: string[] | undefined
	environmentNames: string[] | undefined

	trueAvatar: Avatar | null

	estimatedAvatar: Avatar | null
	reductionPercentage: number

	actions: Actions
}

export const useAvatars = create<Store>(
	immerStore((set, get) => ({
		distanceType: "far",

		trueAvatar: null,
		estimatedAvatar: null,

		avatarName: undefined,
		environmentName: undefined,
		avatarNames: undefined,
		environmentNames: undefined,

		reductionPercentage: 0.5,

		url: "http://127.0.0.1:5000",

		actions: {
			loadTrueAvatar: (callback) => {
				const { url, avatarName, distanceType } = get()
				if (avatarName) {
					fetch(`${url}/load/true/${distanceType}/${avatarName}`)
						.then((response) => response.json())
						.then((data) => {
							set((state) => {
								state.trueAvatar = {
									points: data.points,
									colors: data.colors,
									normals: data.normals,
									distances: data.distances,
								}
							})
							if (callback) callback()
						})
				}
			},

			loadEstimatedAvatar: (callback) => {
				const { url, avatarName, environmentName, distanceType } = get()
				if (avatarName && environmentName) {
					fetch(`${url}/load/estimated/${distanceType}/${avatarName}/${environmentName}`)
						.then((response) => response.json())
						.then((data) => {
							set((state) => {
								state.estimatedAvatar = {
									points: data.points,
									colors: data.colors,
									normals: data.normals,
									distances: data.distances,
								}
							})
							if (callback) callback()
						})
				}
			},
			setAvatarName: (avatarName) => {
				set((state) => {
					state.avatarName = avatarName
				})
			},
			setDistanceType: (distanceType) => {
				set((state) => {
					state.distanceType = distanceType
				})
			},
			setEnvironmentName: (environmentName) => {
				set((state) => {
					state.environmentName = environmentName
				})
			},
			setReductionPercentage: (reductionPercentage) => {
				set((state) => {
					state.reductionPercentage = reductionPercentage
				})
			},
			getNames: () => {
				const { url, actions } = get()
				fetch(`${url}/load/names`)
					.then((response) => response.json())
					.then((data) => {
						const sortedAvatarNames = data.avatars.sort()
						const sortedEnvironmentNames = data.environments.sort()

						set((state) => {
							state.avatarNames = sortedAvatarNames
							state.environmentNames = sortedEnvironmentNames
							state.avatarName = sortedAvatarNames[0]
							state.environmentName = sortedEnvironmentNames[0]
						})
					})
			},
		},
	}))
)

useAvatars.getState().actions.setAvatarName("Female_Adult_01")
useAvatars.getState().actions.loadTrueAvatar(undefined)
useAvatars.getState().actions.getNames()
