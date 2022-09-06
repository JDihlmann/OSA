import Head from "next/head"
import { AppProps } from "next/app"
import "tailwindcss/tailwind.css"
import "../styles/globals.css"

export default function MyApp({ Component, pageProps }: AppProps) {
	return (
		<>
			<Head>
				<meta charSet="utf-8" />
				<title> Image Cloud </title>
			</Head>
			<Component {...pageProps} />
		</>
	)
}
