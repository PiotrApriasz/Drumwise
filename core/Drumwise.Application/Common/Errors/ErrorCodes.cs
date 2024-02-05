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
    }
}