﻿//------------------------------------------------------------------------------
// <auto-generated>
//     This code was generated by a tool.
//
//     Changes to this file may cause incorrect behavior and will be lost if
//     the code is regenerated.
// </auto-generated>
//------------------------------------------------------------------------------

namespace Drumwise.Application.Common.Errors {
    using System;
    
    
    [System.CodeDom.Compiler.GeneratedCodeAttribute("System.Resources.Tools.StronglyTypedResourceBuilder", "4.0.0.0")]
    [System.Diagnostics.DebuggerNonUserCodeAttribute()]
    [System.Runtime.CompilerServices.CompilerGeneratedAttribute()]
    internal class ErrorMessages {
        
        private static System.Resources.ResourceManager resourceMan;
        
        private static System.Globalization.CultureInfo resourceCulture;
        
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1811:AvoidUncalledPrivateCode")]
        internal ErrorMessages() {
        }
        
        [System.ComponentModel.EditorBrowsableAttribute(System.ComponentModel.EditorBrowsableState.Advanced)]
        internal static System.Resources.ResourceManager ResourceManager {
            get {
                if (object.Equals(null, resourceMan)) {
                    System.Resources.ResourceManager temp = new System.Resources.ResourceManager("Drumwise.Application.Common.Errors.ErrorMessages", typeof(ErrorMessages).Assembly);
                    resourceMan = temp;
                }
                return resourceMan;
            }
        }
        
        [System.ComponentModel.EditorBrowsableAttribute(System.ComponentModel.EditorBrowsableState.Advanced)]
        internal static System.Globalization.CultureInfo Culture {
            get {
                return resourceCulture;
            }
            set {
                resourceCulture = value;
            }
        }
        
        internal static string Identity_ExperienceBelowZero {
            get {
                return ResourceManager.GetString("Identity.ExperienceBelowZero", resourceCulture);
            }
        }
        
        internal static string Identity_UserNotFound {
            get {
                return ResourceManager.GetString("Identity.UserNotFound", resourceCulture);
            }
        }
        
        internal static string Identity_SurnameIsRequired {
            get {
                return ResourceManager.GetString("Identity.SurnameIsRequired", resourceCulture);
            }
        }
        
        internal static string Identity_NameIsRequired {
            get {
                return ResourceManager.GetString("Identity.NameIsRequired", resourceCulture);
            }
        }
        
        internal static string Identity_RoleIsRequired {
            get {
                return ResourceManager.GetString("Identity.RoleIsRequired", resourceCulture);
            }
        }
        
        internal static string Identity_UnknownRole {
            get {
                return ResourceManager.GetString("Identity.UnknownRole", resourceCulture);
            }
        }
        
        internal static string Homework_TitleIsRequired {
            get {
                return ResourceManager.GetString("Homework.TitleIsRequired", resourceCulture);
            }
        }
        
        internal static string Homework_ToLittleDeadline {
            get {
                return ResourceManager.GetString("Homework.ToLittleDeadline", resourceCulture);
            }
        }
        
        internal static string Homework_AssignedToIsRequired {
            get {
                return ResourceManager.GetString("Homework.AssignedToIsRequired", resourceCulture);
            }
        }
        
        internal static string Identity_InvalidEmail {
            get {
                return ResourceManager.GetString("Identity.InvalidEmail", resourceCulture);
            }
        }
        
        internal static string Homework_NotFound {
            get {
                return ResourceManager.GetString("Homework.NotFound", resourceCulture);
            }
        }
        
        internal static string DrumsAudio_IncorrectDrumsAudio {
            get {
                return ResourceManager.GetString("DrumsAudio.IncorrectDrumsAudio", resourceCulture);
            }
        }
        
        internal static string DrumsAudio_UploadingFailed {
            get {
                return ResourceManager.GetString("DrumsAudio.UploadingFailed", resourceCulture);
            }
        }
    }
}
