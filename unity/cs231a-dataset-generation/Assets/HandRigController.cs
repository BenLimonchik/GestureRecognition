using System;
using System.Reflection;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public class HandRigController : MonoBehaviour
{
	[System.Serializable]
	public class JointAngle
	{
		public Transform Transform;
		public Vector3 InitialAngles;
		public Vector3 Angles;
	}

	[System.Serializable]
	public class Finger
	{
		public JointAngle Palm;
		public JointAngle Joint1;
		public JointAngle Joint2;
		public JointAngle Joint3;
	}

	[System.Serializable]
	public class Thumb
	{
		public JointAngle Palm;
		public JointAngle Joint1;
		public JointAngle Joint2;
	}

	[System.Serializable]
	public class Hand
	{
		public JointAngle root;
		public Thumb thumb;
		public Finger index;
		public Finger middle;
		public Finger ring;
		public Finger pinky;
	}

	public Hand hand;

	private JointAngle[] joints;

	private void Start ()
	{
		joints = new JointAngle[]{
			hand.root,
			hand.thumb.Palm,
			hand.thumb.Joint1,
			hand.thumb.Joint2,
			hand.index.Palm,
			hand.index.Joint1,
			hand.index.Joint2,
			hand.index.Joint3,
			hand.middle.Palm,
			hand.middle.Joint1,
			hand.middle.Joint2,
			hand.middle.Joint3,
			hand.ring.Palm,
			hand.ring.Joint1,
			hand.ring.Joint2,
			hand.ring.Joint3,
			hand.pinky.Palm,
			hand.pinky.Joint1,
			hand.pinky.Joint2,
			hand.pinky.Joint3,
		};

		Debug.Log("Saving initial angles of joints.");
		ForEachJointAngle((JointAngle joint) => {
			joint.InitialAngles = joint.Transform.localEulerAngles;
			joint.Angles = joint.InitialAngles;
		});
	}

	public void Update ()
	{
		Debug.Log("Updating local euler angles.");
		ForEachJointAngle((JointAngle joint) => {
			joint.Transform.localEulerAngles = joint.Angles;
		});
	}

	private void ForEachJointAngle(Action<JointAngle> lambda) {
		foreach (var joint in joints)
		{
			lambda(joint);
		}
	}
}
