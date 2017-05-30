using System.Text;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ImageCaptureController : MonoBehaviour {
	public int numScreenshots = 0;
	public string fileName = "derp";
	public string fileExtension = "jpg";

	public Material backgroundMaterial;
	public int numBackgrounds = 1;

	public Light sunLight;
	public float maxSunRotX, maxSunRotY, maxSunRotZ;
	public float lightIntensityMin = 0.0f;
	public float lightIntensityMax = 1.0f;

	public Transform hand;
	public float maxDeltaX, maxDeltaY, maxDeltaZ, maxRotX, maxRotY, maxRotZ, minScale, maxScale;

	private Vector3 sunInitialRotation;
	private Vector3 handInitialPosition;
	private Vector3 handInitialRotation;

	private string outputName = "";
	private int i = 0;
	private Color averageBgColor = Color.black;
	private int[] backgroundOrder;

	void Start() {
		int ticks = System.Environment.TickCount;
		Debug.Log("Ticks => seed = " + ticks);
		Random.seed = ticks;

		sunInitialRotation = sunLight.transform.eulerAngles;
		handInitialPosition = hand.position;
		handInitialRotation = hand.eulerAngles;

		backgroundOrder = new int[numBackgrounds];
		StringBuilder sb = new StringBuilder();
		sb.Append("[");
		for (int j = 0; j < numBackgrounds; ++j) {
			int numBgsLeft = numBackgrounds - j;
			backgroundOrder[j] = Random.Range(1, numBgsLeft);
			sb.Append(backgroundOrder[j].ToString());
			sb.Append(",");
		}
		sb.Append("]");
		Debug.Log(sb.ToString());
	}

	void OnPreRender() {
		if (i < numScreenshots) {
			SetupBackground();
			SetupLighting();
			SetupHandModel();
			outputName = string.Format("{0}{1}.{2}", fileName, (i + 1).ToString("D3"), fileExtension);
			++i;
		}
	}

	void OnPostRender() {
		if (outputName.Length > 0) {
			Application.CaptureScreenshot(outputName);
			outputName = "";
		}
	}

	private void SetupBackground() {
		string texturePath = string.Format("Backgrounds/pic_{0}", backgroundOrder[i].ToString("D3"));
		Debug.Log(texturePath);
		Texture2D texture = Resources.Load(texturePath) as Texture2D;
		averageBgColor = GetAverageColor(texture);
		backgroundMaterial.mainTexture = texture;
		backgroundMaterial.mainTextureOffset = new Vector2(Random.value, Random.value);
	}

	private void SetupLighting() {
		float rotX = Random.Range(-maxRotX, maxRotX);
		float rotY = Random.Range(-maxRotX, maxRotX);
		float rotZ = Random.Range(-maxRotX, maxRotX);

		sunLight.transform.eulerAngles = sunInitialRotation + new Vector3(rotX, rotY, rotZ);

		Debug.Assert(lightIntensityMin < lightIntensityMax);
		sunLight.intensity = (lightIntensityMax - lightIntensityMin) * Random.value + lightIntensityMin;
		sunLight.color = averageBgColor;

		Debug.LogFormat("Average BG color [{0}, {1}, {2}]", averageBgColor.r, averageBgColor.g, averageBgColor.b);
	}

	private void SetupHandModel() {
		float deltaX = Random.Range(-maxDeltaX, maxDeltaX);
		float deltaY = Random.Range(-maxDeltaY, maxDeltaY);
		float deltaZ = Random.Range(-maxDeltaZ, maxDeltaZ);
		float rotX = Random.Range(-maxRotX, maxRotX);
		float rotY = Random.Range(-maxRotX, maxRotX);
		float rotZ = Random.Range(-maxRotX, maxRotX);
		float scale = Random.Range(minScale, maxScale);

		hand.position = handInitialPosition + new Vector3(deltaX, deltaY, deltaZ);
		hand.eulerAngles = handInitialRotation + new Vector3(rotX, rotY, rotZ);
		hand.localScale = new Vector3(scale, scale, scale);
	}

	private Color GetAverageColor(Texture2D tex) {
		Color[] pixels = tex.GetPixels();

		Color sum = new Color();
		for (int i = 0; i < pixels.Length; ++i) {
			sum += pixels[i];
		}

		sum /= pixels.Length;
		return sum;
	}
}
