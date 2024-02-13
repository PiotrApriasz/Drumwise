namespace Drumwise.Application.Common.Errors;

public static class ErrorCodes
{
    public static class Identity
    {
        public const string UserNotFound = "Identity.UserNotFound";
        public const string ExperienceBelowZero = "Identity.ExperienceBelowZero";
        public const string NameIsRequired = "Identity.NameIsRequired";
        public const string SurnameIsRequired = "Identity.SurnameIsRequired";
        public const string RoleIsRequired = "Identity.RoleIsRequired";
        public const string UnknownRole = "Identity.UnknownRole";
        public const string InvalidEmail = "Identity.InvalidEmail";
    }

    public static class Homework
    {
        public const string TitleIsRequired = "Homework.TitleIsRequired";
        public const string ToLittleDeadline = "Homework.ToLittleDeadline";
        public const string AssignedToIsRequired = "Homework.AssignedToIsRequired";
    }
}