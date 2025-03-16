namespace Drumwise.Application.Common.Security;

[AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
public class AuthorizeAttribute : Attribute
{
    public string Roles { get; set; } = string.Empty;
    public string Policy { get; set; } = string.Empty;
}