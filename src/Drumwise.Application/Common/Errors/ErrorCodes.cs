namespace Drumwise.Application.Common.Errors;

public static class ErrorCodes
{
    public static class Identity
    {
        public const string RegisterAdditionalUserDataError = "Identity.RegisterAdditionalUserDataError";
        
        public const string UserNotFound = "Identity.UserNotFound";
        public const string ExperienceBelowZero = "Identity.ExperienceBelowZero";
    }
}